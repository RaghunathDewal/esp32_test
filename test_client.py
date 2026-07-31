"""
Simulates an ESP32 client against the bridge server, using your PC's
mic/speakers instead of real ESP32 hardware. Use this to confirm the
server + Gemini Live integration works BEFORE flashing ESP32 firmware.

Usage:
    python test_client.py wss://<your-app>.onrender.com/ws
    python test_client.py ws://localhost:8000/ws   # against a local server
"""

import asyncio
import sys

import pyaudio
import websockets

FORMAT = pyaudio.paInt16
CHANNELS = 1
SEND_SAMPLE_RATE = 16000
RECEIVE_SAMPLE_RATE = 24000
CHUNK_SIZE = 1024

pya = pyaudio.PyAudio()


async def mic_to_server(ws, stream):
    while True:
        data = await asyncio.to_thread(
            stream.read, CHUNK_SIZE, exception_on_overflow=False
        )
        await ws.send(data)


async def server_to_speaker(ws, out_stream):
    async for message in ws:
        if isinstance(message, (bytes, bytearray)):
            await asyncio.to_thread(out_stream.write, message)


async def main(url: str):
    in_stream = pya.open(
        format=FORMAT,
        channels=CHANNELS,
        rate=SEND_SAMPLE_RATE,
        input=True,
        frames_per_buffer=CHUNK_SIZE,
    )
    out_stream = pya.open(
        format=FORMAT,
        channels=CHANNELS,
        rate=RECEIVE_SAMPLE_RATE,
        output=True,
        frames_per_buffer=CHUNK_SIZE,
    )

    print(f"Connecting to {url} ...")
    async with websockets.connect(url) as ws:
        print("Connected. Speak into your mic (Ctrl+C to stop).")
        await asyncio.gather(
            mic_to_server(ws, in_stream),
            server_to_speaker(ws, out_stream),
        )


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python test_client.py <ws:// or wss:// server URL>")
        sys.exit(1)
    try:
        asyncio.run(main(sys.argv[1]))
    except KeyboardInterrupt:
        pass
