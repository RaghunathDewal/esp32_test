"""Run: python test_resampler.py  (no framework needed)"""
import math
from array import array

from server import Resampler24to16

# 1s of 440Hz tone at 24kHz
pcm = array("h", (int(10000 * math.sin(2 * math.pi * 440 * i / 24000)) for i in range(24000))).tobytes()

whole = Resampler24to16().process(pcm)

# same audio fed in awkward chunks, including odd-byte and 2-byte ones
r, parts, i = Resampler24to16(), [], 0
for size in (2, 7, 1920, 3841, 15360, 1):
    while i < len(pcm):
        parts.append(r.process(pcm[i:i + size]))
        i += size
        break
    if i >= len(pcm):
        break
while i < len(pcm):
    parts.append(r.process(pcm[i:i + 4001]))
    i += 4001
chunked = b"".join(parts)

assert abs(len(whole) // 2 - 16000) <= 2, len(whole) // 2
assert whole == chunked, "chunk boundaries changed the output"
print(f"OK: {len(pcm)//2} samples @24k -> {len(whole)//2} samples @16k, chunk-invariant")
