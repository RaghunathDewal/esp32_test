"""
Simulates an ESP32 client against the bridge server, using your PC's
mic/speakers instead of real ESP32 hardware. Use this to confirm the
server + Gemini Live integration works BEFORE flashing ESP32 firmware.

Usage:
    python test_client.py wss://<your-app>.onrender.com/ws
    python test_client.py ws://localhost:8000/ws   # against a local server
    python test_client.py wss://<your-app>.onrender.com/ws recording.wav
        # sends a WAV file instead of the mic (16kHz mono 16-bit PCM)
"""

import asyncio
import sys
import wave

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


async def wav_to_server(ws, wav_path: str):
    """Streams a WAV file to the server at real-time pace, mimicking a live
    mic feed (dumping it all at once would starve Gemini's VAD of realistic
    timing)."""
    with wave.open(wav_path, "rb") as wf:
        if (wf.getframerate(), wf.getnchannels(), wf.getsampwidth()) != (SEND_SAMPLE_RATE, CHANNELS, 2):
            print(
                f"Warning: {wav_path} is {wf.getframerate()}Hz "
                f"{wf.getnchannels()}ch {wf.getsampwidth() * 8}-bit — "
                f"expected {SEND_SAMPLE_RATE}Hz {CHANNELS}ch 16-bit. "
                "Sending anyway, but Gemini may mishear it."
            )
        chunk_seconds = CHUNK_SIZE / SEND_SAMPLE_RATE
        while data := wf.readframes(CHUNK_SIZE):
            await ws.send(data)
            await asyncio.sleep(chunk_seconds)

    # Gemini's VAD detects end-of-speech from a trailing quiet period in the
    # audio stream itself (see send_realtime_input docs) — a live mic gives
    # it that for free via room silence, a WAV file just stops. Fake it,
    # otherwise Gemini never realizes you're done talking and never replies.
    silence_chunk = b"\x00" * (CHUNK_SIZE * 2)
    for _ in range(int(1.5 / chunk_seconds)):
        await ws.send(silence_chunk)
        await asyncio.sleep(chunk_seconds)

    print("WAV file fully sent — waiting for Gemini's reply "
          "(Ctrl+C to stop after it plays back).")
    await asyncio.Event().wait()  # keep the connection open for the reply


async def receive_from_server(ws, playback_queue: asyncio.Queue):
    """Pulls messages off the socket as fast as they arrive and hands them
    to the playback queue. Kept separate from playback so a slow speaker
    write never blocks us from noticing a turn-complete/interrupt signal."""
    async for message in ws:
        if isinstance(message, (bytes, bytearray)):
            print(f"[RX] audio chunk: {len(message)} bytes")
            await playback_queue.put(message)
        elif message == "__TURN_COMPLETE__":
            print("[RX] __TURN_COMPLETE__")
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


async def main(url: str, wav_path: str | None):
    out_stream = pya.open(
        format=FORMAT,
        channels=CHANNELS,
        rate=RECEIVE_SAMPLE_RATE,
        output=True,
        frames_per_buffer=CHUNK_SIZE,
    )

    print(f"Connecting to {url} ...")
    async with websockets.connect(url) as ws:
        playback_queue: asyncio.Queue = asyncio.Queue()

        if wav_path:
            print(f"Connected. Sending {wav_path} ...")
            send_task = wav_to_server(ws, wav_path)
        else:
            in_stream = pya.open(
                format=FORMAT,
                channels=CHANNELS,
                rate=SEND_SAMPLE_RATE,
                input=True,
                frames_per_buffer=CHUNK_SIZE,
            )
            print("Connected. Speak into your mic (Ctrl+C to stop).")
            send_task = mic_to_server(ws, in_stream)

        await asyncio.gather(
            send_task,
            receive_from_server(ws, playback_queue),
            play_from_queue(playback_queue, out_stream),
        )


if __name__ == "__main__":
    if len(sys.argv) not in (2, 3):
        print("Usage: python test_client.py <ws:// or wss:// server URL> [file.wav]")
        sys.exit(1)
    try:
        asyncio.run(main(sys.argv[1], sys.argv[2] if len(sys.argv) == 3 else None))
    except KeyboardInterrupt:
        pass