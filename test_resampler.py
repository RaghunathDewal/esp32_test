"""Run: python test_resampler.py  (no framework needed)"""
import math
import os
from array import array
from itertools import cycle

os.environ.setdefault("GEMINI_API_KEY", "offline-test-key")
from server import Resampler24kHz

# 1s of 440Hz tone at 24kHz
pcm = array("h", (int(10000 * math.sin(2 * math.pi * 440 * i / 24000)) for i in range(24000))).tobytes()


def check(target):
    whole = Resampler24kHz(target).process(pcm)

    assert len(whole) // 2 == target, len(whole) // 2
    if target == 8000:
        assert whole == array("h", array("h", pcm)[::3]).tobytes()

    # Include byte-sized chunks and empty calls while waiting for input.
    for sizes in ((1,), (2,), (4,), (7,), (2, 7, 1920, 3841, 15360, 1, 4001)):
        r, parts, i = Resampler24kHz(target), [], 0
        for size in cycle(sizes):
            if i >= len(pcm):
                break
            parts.append(r.process(pcm[i:i + size]))
            parts.append(r.process(b""))
            i += size
        assert whole == b"".join(parts), f"chunk boundaries changed output for target={target}, sizes={sizes}"
    print(f"OK: {len(pcm)//2} samples @24k -> {len(whole)//2} samples @{target}, chunk-invariant")


check(8000)
check(16000)
