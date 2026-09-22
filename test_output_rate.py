"""Offline regression coverage for the WebSocket output-rate contract."""

import asyncio
import math
import os
import struct
import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock, patch

os.environ.setdefault("GEMINI_API_KEY", "offline-test-key")

import server


class OutputRateTest(unittest.IsolatedAsyncioTestCase):
    async def test_output_rates_keep_microphone_at_16khz(self):
        pcm = struct.pack(
            "<24000h",
            *(int(12000 * math.sin(2 * math.pi * 440 * i / 24000))
              for i in range(24000)),
        )
        chunks = [pcm[i:i + 4096] for i in range(0, len(pcm), 4096)]
        microphone = pcm[:640]
        cases = (
            ("8000", 24000),  # Unsupported URL rate falls back to 24 kHz.
            ("16000", 8000),  # Firmware URL alias, not 16 kHz output.
            ("24000", 24000),
            (None, 24000),
            ("invalid", 24000),
            ("0", 24000),
            ("22050", 24000),
            ("48000", 24000),
        )
        for requested, expected_rate in cases:
            with self.subTest(output_rate=requested):
                turn_complete = asyncio.Event()
                microphone_sent = asyncio.Event()

                async def receive_bytes():
                    if not microphone_sent.is_set():
                        return microphone
                    await turn_complete.wait()
                    raise server.WebSocketDisconnect(code=1000)

                async def send_input(*, audio):
                    microphone_sent.set()

                async def receive():
                    await microphone_sent.wait()
                    # Wait for cancellation while the WebSocket reports disconnect.
                    if turn_complete.is_set():
                        await asyncio.Future()
                    for chunk in chunks:
                        yield SimpleNamespace(
                            data=chunk, tool_call=None, server_content=None,
                        )

                async def send_text(text):
                    if text == "__TURN_COMPLETE__":
                        turn_complete.set()

                websocket = SimpleNamespace(
                    query_params={} if requested is None else {"output_rate": requested},
                    accept=AsyncMock(),
                    close=AsyncMock(),
                    receive_bytes=AsyncMock(side_effect=receive_bytes),
                    send_bytes=AsyncMock(),
                    send_text=AsyncMock(side_effect=send_text),
                )
                session = SimpleNamespace(
                    receive=receive,
                    send_realtime_input=AsyncMock(side_effect=send_input),
                )
                connection = AsyncMock()
                connection.__aenter__.return_value = session
                client = Mock()
                client.aio.live.connect.return_value = connection

                with patch.object(server, "client", client), patch.object(server, "logger"):
                    await asyncio.wait_for(server.websocket_endpoint(websocket), timeout=2)

                websocket.accept.assert_awaited_once()
                websocket.close.assert_not_awaited()
                websocket.send_text.assert_awaited_once_with("__TURN_COMPLETE__")
                client.aio.live.connect.assert_called_once_with(
                    model=server.MODEL, config=server.CONFIG,
                )
                session.send_realtime_input.assert_awaited_once()
                audio = session.send_realtime_input.await_args.kwargs["audio"]
                self.assertEqual(audio.data, microphone)
                self.assertEqual(audio.mime_type, "audio/pcm;rate=16000")

                frames = [call.args[0] for call in websocket.send_bytes.await_args_list]
                self.assertEqual(sum(map(len, frames)), expected_rate * 2)
                if expected_rate == 24000:
                    expected_frames = chunks
                else:
                    resampler = server.Resampler24kHz(expected_rate)
                    expected_frames = [resampler.process(chunk) for chunk in chunks]
                    expected_frames = [frame for frame in expected_frames if frame]
                self.assertEqual(frames, expected_frames)


if __name__ == "__main__":
    unittest.main()
