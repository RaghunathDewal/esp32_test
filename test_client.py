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


async def receive_from_server(ws, playback_queue: asyncio.Queue):
    """Pulls messages off the socket as fast as they arrive and hands them
    to the playback queue. Kept separate from playback so a slow speaker
    write never blocks us from noticing a turn-complete/interrupt signal."""
    async for message in ws:
        if isinstance(message, (bytes, bytearray)):
            await playback_queue.put(message)
        elif message == "__TURN_COMPLETE__":
            # Gemini's turn ended. Any audio still sitting in the queue
            # belongs to a reply that's already finished; if a new turn's
            # audio starts landing next, draining old + new together is
            # what causes the "repeats with lag" effect. Flush it.
            while not playback_queue.empty():
                playback_queue.get_nowait()


async def play_from_queue(playback_queue: asyncio.Queue, out_stream):
    while True:
        chunk = await playback_queue.get()
        await asyncio.to_thread(out_stream.write, chunk)


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
        playback_queue: asyncio.Queue = asyncio.Queue()
        await asyncio.gather(
            mic_to_server(ws, in_stream),
            receive_from_server(ws, playback_queue),
            play_from_queue(playback_queue, out_stream),
        )


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python test_client.py <ws:// or wss:// server URL>")
        sys.exit(1)
    try:
        asyncio.run(main(sys.argv[1]))
    except KeyboardInterrupt:
        pass