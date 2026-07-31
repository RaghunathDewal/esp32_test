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
from google import genai
from google.genai import types
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
    tools=TOOLS,
)


@app.get("/")
async def health():
    """Simple health check endpoint Render can ping."""
    return {"status": "ok", "service": "gemini-esp32-bridge"}


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    logger.info("ESP32 client connected")

    if not GEMINI_API_KEY:
        await websocket.close(code=1011, reason="Server missing GEMINI_API_KEY")
        return

    try:
        async with client.aio.live.connect(model=MODEL, config=CONFIG) as session:

            async def esp32_to_gemini():
                """Read binary audio frames from the ESP32 and forward to Gemini."""
                while True:
                    data = await websocket.receive_bytes()
                    await session.send_realtime_input(
                        audio=types.Blob(data=data, mime_type="audio/pcm")
                    )

            async def gemini_to_esp32():
                """Read Gemini's audio/tool output and forward audio to the ESP32."""
                while True:
                    turn = session.receive()
                    async for response in turn:
                        if data := response.data:
                            await websocket.send_bytes(data)
                            continue
                        if text := response.text:
                            logger.info("Gemini text: %s", text)
                        if tool_call := response.tool_call:
                            await handle_tool_call(session, tool_call)

                    # Gemini signals end-of-turn here. If the user interrupts
                    # mid-reply, a new turn starts before the old one drains —
                    # tell the client to flush anything still queued so old
                    # audio doesn't overlap/repeat under the new reply.
                    await websocket.send_text("__TURN_COMPLETE__")

            await asyncio.gather(esp32_to_gemini(), gemini_to_esp32())

    except WebSocketDisconnect:
        logger.info("ESP32 client disconnected")
    except Exception:
        logger.exception("Error in websocket session")
        try:
            await websocket.close(code=1011, reason="Internal error")
        except Exception:
            pass


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

    await session.send_tool_response(function_responses=function_responses)


if __name__ == "__main__":
    import uvicorn

    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)