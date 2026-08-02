"""
WebSocket bridge server: ESP32  <-->  this server  <-->  Gemini Live API.

Deploy this on Render (or any host) as a small always-on WebSocket service.
Your ESP32 firmware connects to  wss://<your-render-app>.onrender.com/ws
and streams raw 16-bit PCM mono audio (16kHz) as binary WebSocket frames.

This server:
  1. Accepts the ESP32 WebSocket connection.
  2. Opens a Gemini Live session per connection.
  3. Forwards every binary frame it receives from the ESP32 -> Gemini
     (as realtime audio input).
  4. Forwards every audio chunk it receives from Gemini -> the ESP32
     (as binary frames, 24kHz PCM16, for the ESP32 to play back).
  5. Runs the same test tools (get_current_time, add_numbers) as the
     desktop script, so you can sanity-check tool calling end-to-end.

This is a TESTING-LEVEL server: no auth, no reconnection hardening,
no rate limiting. Add those before using it for anything real.
"""

import asyncio
import json
import logging
import os

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse
from google import genai
from google.genai import types
from websockets.exceptions import ConnectionClosedError
from dotenv import load_dotenv
load_dotenv() 
from tools import TOOLS, TOOL_IMPLEMENTATIONS

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("gemini-esp32-bridge")

MODEL = "models/gemini-3.1-flash-live-preview"
SEND_SAMPLE_RATE = 16000  # audio coming FROM the ESP32 mic
RECEIVE_SAMPLE_RATE = 24000  # audio going TO the ESP32 speaker

GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")

app = FastAPI()

client = genai.Client(
    http_options={"api_version": "v1beta"},
    api_key=GEMINI_API_KEY,
)

CONFIG = types.LiveConnectConfig(
    response_modalities=["AUDIO"],
    media_resolution="MEDIA_RESOLUTION_MEDIUM",
    speech_config=types.SpeechConfig(
        voice_config=types.VoiceConfig(
            prebuilt_voice_config=types.PrebuiltVoiceConfig(voice_name="Zephyr")
        )
    ),
    # Gemini transcribes the user's spoken audio and sends the text back
    # alongside the audio reply — this is what lets us log what the user
    # asked, without doing our own speech-to-text.
    input_audio_transcription=types.AudioTranscriptionConfig(),
    tools=TOOLS,
)


@app.get("/")
async def health():
    """Simple health check endpoint Render can ping."""
    return {"status": "ok", "service": "gemini-esp32-bridge"}


@app.get("/console")
async def console():
    """
    Serves the browser test console over HTTPS. Needed for testing from
    iOS Safari (and mobile browsers generally) since getUserMedia (mic
    access) requires a secure context — a local file:// page won't get
    mic permission, but this route does since it's served over https://.
    """
    static_path = os.path.join(os.path.dirname(__file__), "static", "console.html")
    return FileResponse(static_path, media_type="text/html")


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    logger.info("ESP32 client connected")

    if not GEMINI_API_KEY:
        await websocket.close(code=1011, reason="Server missing GEMINI_API_KEY")
        return

    MAX_RECONNECTS = 5
    reconnect_count = 0

    while reconnect_count <= MAX_RECONNECTS:
        try:
            async with client.aio.live.connect(model=MODEL, config=CONFIG) as session:
                if reconnect_count > 0:
                    logger.info(
                        "Reconnected to Gemini (attempt %d)", reconnect_count
                    )

                async def esp32_to_gemini():
                    """Read binary audio frames from the ESP32 and forward to Gemini."""
                    while True:
                        data = await websocket.receive_bytes()
                        await session.send_realtime_input(
                            audio=types.Blob(
                                data=data,
                                mime_type=f"audio/pcm;rate={SEND_SAMPLE_RATE}",
                            )
                        )

                async def gemini_to_esp32():
                    """Read Gemini's audio/tool output and forward audio to the ESP32.

                    IMPORTANT: session.receive() is a single continuous stream —
                    it must be iterated once, not recreated in a loop. Turn
                    boundaries are detected via server_content.turn_complete,
                    not by the iterator ending (it doesn't end between turns).
                    """
                    async for response in session.receive():
                        try:
                            if data := response.data:
                                await websocket.send_bytes(data)
                                continue
                            if text := response.text:
                                logger.info("Gemini text: %s", text)
                            if tool_call := response.tool_call:
                                await handle_tool_call(session, tool_call)

                            server_content = getattr(response, "server_content", None)
                            if server_content is not None:
                                input_transcript = getattr(
                                    server_content, "input_transcription", None
                                )
                                if input_transcript is not None and input_transcript.text:
                                    logger.info("USER SAID: %s", input_transcript.text)

                                if getattr(server_content, "turn_complete", False):
                                    # Tell the client to flush any stale queued
                                    # audio so an interruption's new turn doesn't
                                    # overlap with the previous one's leftovers.
                                    await websocket.send_text("__TURN_COMPLETE__")
                                    logger.info("Turn complete.")
                        except Exception:
                            # A parsing quirk in one response (e.g. an
                            # unexpected field shape) must not kill the whole
                            # receive loop — that would silently stop us from
                            # ever hearing the user again after one turn.
                            logger.exception(
                                "Error handling one response — continuing loop"
                            )

                sender = asyncio.create_task(esp32_to_gemini())
                receiver = asyncio.create_task(gemini_to_esp32())

                # If either side dies (client disconnects, Gemini closes the
                # session, etc.) the other must stop too — otherwise one task
                # keeps trying to read/write a session that's already gone,
                # which is its own source of confusing errors.
                done, pending = await asyncio.wait(
                    [sender, receiver], return_when=asyncio.FIRST_EXCEPTION
                )
                for task in pending:
                    task.cancel()
                await asyncio.gather(*pending, return_exceptions=True)

                # Re-raise whatever actually failed, so we can tell apart
                # "ESP32 disconnected" (stop entirely) from "Gemini's
                # connection dropped" (worth reconnecting to Gemini only).
                for task in done:
                    if task.exception():
                        raise task.exception()

            # Session ended cleanly (no exception) — nothing more to do.
            return

        except WebSocketDisconnect:
            logger.info("ESP32 client disconnected")
            return
        except ConnectionClosedError as e:
            # This is Gemini's side dropping (e.g. keepalive ping timeout),
            # not the ESP32/browser client. Reconnect to Gemini and keep
            # the client's WebSocket open rather than forcing a full
            # client-side reconnect over a flaky hop.
            reconnect_count += 1
            logger.warning(
                "Gemini connection dropped (%s) — reconnecting (%d/%d)",
                e, reconnect_count, MAX_RECONNECTS,
            )
            if reconnect_count > MAX_RECONNECTS:
                logger.error("Max Gemini reconnect attempts exceeded, giving up")
                try:
                    await websocket.close(code=1011, reason="Gemini connection unstable")
                except Exception:
                    pass
                return
            continue
        except Exception as e:
            logger.exception("Error in websocket session")
            # google.genai's APIError often has empty response_json for Live
            # API close codes — log every attribute that might carry detail
            # so we're not stuck with just "1008 None" next time this happens.
            for attr in ("code", "status_code", "reason", "response_json", "message", "args"):
                if hasattr(e, attr):
                    logger.error("  %s = %r", attr, getattr(e, attr))
            try:
                await websocket.close(code=1011, reason="Internal error")
            except Exception:
                pass
            return


async def handle_tool_call(session, tool_call):
    """Executes each requested function call and sends the results back to Gemini."""
    function_responses = []
    for fc in tool_call.function_calls:
        logger.info("Tool call: %s(%s)", fc.name, fc.args)
        impl = TOOL_IMPLEMENTATIONS.get(fc.name)
        if impl is None:
            result = {"error": f"Unknown tool: {fc.name}"}
        else:
            try:
                result = impl(**(fc.args or {}))
            except Exception as e:  # noqa: BLE001
                result = {"error": str(e)}
        logger.info("Tool result: %s -> %s", fc.name, json.dumps(result))

        function_responses.append(
            types.FunctionResponse(id=fc.id, name=fc.name, response=result)
        )

    logger.info("Sending tool response...")
    await session.send_tool_response(function_responses=function_responses)
    logger.info("Tool response sent.")


if __name__ == "__main__":
    import uvicorn

    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)