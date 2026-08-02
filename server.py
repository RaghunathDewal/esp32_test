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

    # Event flag: set = safe to send mic audio; cleared = tool call pending, pause mic input
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
                    """Read binary audio frames from the ESP32 and forward to Gemini.

                    Per Google's docs: if the incoming audio stream pauses for more
                    than ~1s, send audio_stream_end to flush Gemini's cached audio,
                    otherwise stale buffer state can interfere with the next utterance.
                    """
                    STREAM_PAUSE_TIMEOUT = 1.0
                    stream_ended = False

                    while True:
                        try:
                            data = await asyncio.wait_for(
                                websocket.receive_bytes(), timeout=STREAM_PAUSE_TIMEOUT
                            )
                        except asyncio.TimeoutError:
                            if not stream_ended:
                                await session.send_realtime_input(audio_stream_end=True)
                                stream_ended = True
                            continue

                        stream_ended = False

                        # Wait if a tool call is currently in progress
                        await can_send_audio.wait()

                        # FIX: the sample rate MUST be in the mime type. Without it
                        # Gemini can't parse the stream reliably and eventually
                        # aborts the session with 1008 "operation was aborted".
                        await session.send_realtime_input(
                            audio=types.Blob(
                                data=data,
                                mime_type=f"audio/pcm;rate={SEND_SAMPLE_RATE}",
                            )
                        )

                async def gemini_to_esp32():
                    """Read Gemini's audio/tool output, log responses, and forward audio to the ESP32."""
                    async for response in session.receive():
                        try:
                            # NOTE: per Google's docs, a single event can carry
                            # MULTIPLE parts at once (e.g. audio AND turn_complete,
                            # or a transcript) with no guaranteed ordering. So every
                            # field below is checked unconditionally — no `continue`
                            # after audio, or we'd silently drop turn_complete when
                            # it arrives bundled with the last audio chunk.

                            # --- Audio out ---
                            if data := response.data:
                                await websocket.send_bytes(data)

                            # --- Text output ---
                            if text := response.text:
                                logger.info("Gemini text response: %s", text)

                            # --- Tool calls ---
                            if tool_call := response.tool_call:
                                logger.info("Gemini requested Tool Call: %s", tool_call)
                                # Pause incoming audio from ESP32 while executing the tool
                                can_send_audio.clear()
                                try:
                                    await handle_tool_call(session, tool_call)
                                finally:
                                    can_send_audio.set()

                            # --- Transcriptions & turn status ---
                            server_content = getattr(response, "server_content", None)
                            if server_content is not None:
                                input_transcript = getattr(
                                    server_content, "input_transcription", None
                                )
                                if input_transcript is not None and input_transcript.text:
                                    logger.info("TRANSCRIPT [User]: %s", input_transcript.text)

                                output_transcript = getattr(
                                    server_content, "output_transcription", None
                                )
                                if output_transcript is not None and output_transcript.text:
                                    logger.info("TRANSCRIPT [Gemini]: %s", output_transcript.text)

                                # User barged in — tell the client to dump queued audio
                                if getattr(server_content, "interrupted", False):
                                    logger.info("Interrupted by user.")
                                    await websocket.send_text("__TURN_COMPLETE__")

                                if getattr(server_content, "turn_complete", False):
                                    logger.info("Gemini turn completed.")
                                    await websocket.send_text("__TURN_COMPLETE__")
                        except Exception:
                            # One malformed/unexpected response must not kill the
                            # whole receive loop — that would silently stop us from
                            # ever hearing the user again after one turn.
                            logger.exception("Error handling one response — continuing")

                sender = asyncio.create_task(esp32_to_gemini())
                receiver = asyncio.create_task(gemini_to_esp32())

                # If either side dies, stop the other too — otherwise one task keeps
                # reading/writing a session that's already gone.
                done, pending = await asyncio.wait(
                    [sender, receiver], return_when=asyncio.FIRST_EXCEPTION
                )
                for task in pending:
                    task.cancel()
                await asyncio.gather(*pending, return_exceptions=True)

                for task in done:
                    if task.exception():
                        raise task.exception()

            # Clean exit, nothing more to do.
            return

        except WebSocketDisconnect:
            logger.info("ESP32 client disconnected")
            return
        except ConnectionClosedError as e:
            # Gemini's side dropped (keepalive timeout, network blip). Reconnect
            # to Gemini while keeping the ESP32's socket open.
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
            can_send_audio.set()  # don't stay paused across a reconnect
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
    """Executes each requested function call and sends the results back to Gemini."""
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

    logger.info("Sending tool response back to Gemini...")
    await session.send_tool_response(function_responses=function_responses)
    logger.info("Tool response sent.")


if __name__ == "__main__":
    import uvicorn

    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
