"""
WebSocket bridge server: ESP32  <-->  this server  <-->  Gemini Live API.
"""

import asyncio
import json
import logging
import os

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from google import genai
from google.genai import types
from websockets.exceptions import ConnectionClosedError
from dotenv import load_dotenv
load_dotenv()
from tools import TOOLS, TOOL_IMPLEMENTATIONS

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("gemini-esp32-bridge")

MODEL = "models/gemini-3.1-flash-live-preview"
SEND_SAMPLE_RATE = 16000      # audio coming FROM the ESP32 mic
RECEIVE_SAMPLE_RATE = 24000   # audio going TO the ESP32 speaker

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
    input_audio_transcription=types.AudioTranscriptionConfig(),
    output_audio_transcription=types.AudioTranscriptionConfig(),
    tools=TOOLS,
)


@app.get("/")
async def health():
    return {"status": "ok", "service": "gemini-esp32-bridge"}


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    logger.info("ESP32 client connected")

    if not GEMINI_API_KEY:
        await websocket.close(code=1011, reason="Server missing GEMINI_API_KEY")
        return

    # set = safe to forward mic audio; cleared = a tool call is in flight
    can_send_audio = asyncio.Event()
    can_send_audio.set()

    MAX_RECONNECTS = 5
    reconnect_count = 0

    while reconnect_count <= MAX_RECONNECTS:
        try:
            async with client.aio.live.connect(model=MODEL, config=CONFIG) as session:
                if reconnect_count > 0:
                    logger.info("Reconnected to Gemini (attempt %d)", reconnect_count)

                async def esp32_to_gemini():
                    """Forward the client's mic audio to Gemini, unmodified."""
                    while True:
                        data = await websocket.receive_bytes()
                        await can_send_audio.wait()
                        await session.send_realtime_input(
                            audio=types.Blob(
                                data=data,
                                mime_type=f"audio/pcm;rate={SEND_SAMPLE_RATE}",
                            )
                        )

                async def gemini_to_esp32():
                    """Forward Gemini's audio to the client, one TURN at a time.

                    session.receive() yields a SINGLE model turn and then the
                    generator terminates (it breaks internally on turn_complete
                    -- see the SDK source). So it must be re-created for each
                    turn inside this outer loop. Dropping the outer loop makes
                    the bridge die silently after the first reply.
                    """
                    while True:
                        turn = session.receive()

                        async for response in turn:
                            try:
                                # A single event can carry several parts at once
                                # (audio + transcript + turn_complete), so every
                                # field is checked -- never `continue` after audio.
                                if data := response.data:
                                    await websocket.send_bytes(data)

                                if tool_call := response.tool_call:
                                    logger.info("Tool call requested: %s", tool_call)
                                    can_send_audio.clear()
                                    try:
                                        await handle_tool_call(session, tool_call)
                                    finally:
                                        can_send_audio.set()

                                sc = getattr(response, "server_content", None)
                                if sc is not None:
                                    it = getattr(sc, "input_transcription", None)
                                    if it is not None and it.text:
                                        logger.info("TRANSCRIPT [User]: %s", it.text)

                                    ot = getattr(sc, "output_transcription", None)
                                    if ot is not None and ot.text:
                                        logger.info("TRANSCRIPT [Gemini]: %s", ot.text)

                                    # Barge-in: flush client playback immediately
                                    # rather than waiting for the turn to wind down.
                                    if getattr(sc, "interrupted", False):
                                        logger.info("Interrupted by user.")
                                        await websocket.send_text("__TURN_COMPLETE__")
                            except Exception:
                                # One odd event must never kill the loop.
                                logger.exception("Error handling one response — continuing")

                        # Generator ended => this turn is done. Tell the client to
                        # flush playback, then loop to receive the NEXT turn.
                        logger.info("Turn complete.")
                        await websocket.send_text("__TURN_COMPLETE__")

                sender = asyncio.create_task(esp32_to_gemini())
                receiver = asyncio.create_task(gemini_to_esp32())

                done, pending = await asyncio.wait(
                    [sender, receiver], return_when=asyncio.FIRST_EXCEPTION
                )
                for task in pending:
                    task.cancel()
                await asyncio.gather(*pending, return_exceptions=True)

                for task in done:
                    if task.exception():
                        raise task.exception()

            return

        except WebSocketDisconnect:
            logger.info("ESP32 client disconnected")
            return
        except ConnectionClosedError as e:
            # Gemini's socket dropped (keepalive timeout / network blip).
            # Reconnect to Gemini while keeping the client's socket open.
            reconnect_count += 1
            logger.warning(
                "Gemini connection dropped (%s) — reconnecting (%d/%d)",
                e, reconnect_count, MAX_RECONNECTS,
            )
            if reconnect_count > MAX_RECONNECTS:
                logger.error("Max Gemini reconnects exceeded, giving up")
                try:
                    await websocket.close(code=1011, reason="Gemini connection unstable")
                except Exception:
                    pass
                return
            can_send_audio.set()
            continue
        except Exception as e:
            logger.exception("Error in websocket session")
            for attr in ("code", "status_code", "reason", "response_json", "message", "args"):
                if hasattr(e, attr):
                    logger.error("  %s = %r", attr, getattr(e, attr))
            try:
                await websocket.close(code=1011, reason="Internal error")
            except Exception:
                pass
            return


async def handle_tool_call(session, tool_call):
    """Execute each requested function call and return the results to Gemini."""
    function_responses = []
    for fc in tool_call.function_calls:
        logger.info("Executing Tool: %s with args: %s", fc.name, fc.args)
        impl = TOOL_IMPLEMENTATIONS.get(fc.name)
        if impl is None:
            result = {"error": f"Unknown tool: {fc.name}"}
        else:
            try:
                result = impl(**(fc.args or {}))
            except Exception as e:  # noqa: BLE001
                result = {"error": str(e)}
        logger.info("Tool Result [%s]: %s", fc.name, json.dumps(result))

        function_responses.append(
            types.FunctionResponse(id=fc.id, name=fc.name, response=result)
        )

    await session.send_tool_response(function_responses=function_responses)
    logger.info("Tool response sent.")


if __name__ == "__main__":
    import uvicorn

    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)