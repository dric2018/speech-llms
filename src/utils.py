import base64

import io


import urllib.request
from typing import Tuple

import numpy as np
import soundfile as sf

def _download_audio_bytes(
        url: str, 
        timeout: int = 30
    ) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return resp.read()


def _read_wav_from_bytes(
        audio_bytes: bytes
    ) -> Tuple[np.ndarray, int]:
    with io.BytesIO(audio_bytes) as f:
        wav, sr = sf.read(f, dtype="float32", always_2d=False)
    return np.asarray(wav, dtype=np.float32), int(sr)


def _to_data_url_base64(
        audio_bytes: bytes, 
        mime: str = "audio/wav"
    ) -> str:
    b64 = base64.b64encode(audio_bytes).decode("utf-8")
    return f"data:{mime};base64,{b64}"


def _print_result(
        title: str,
        results
    ) -> None:
    print(f"\n===== {title} =====")
    for i, r in enumerate(results):
        print(f"[sample {i}] language={r.language!r}")
        print(f"[sample {i}] text={r.text!r}")
        if r.time_stamps is not None and len(r.time_stamps) > 0:
            head = r.time_stamps[0]
            tail = r.time_stamps[-1]
            print(f"[sample {i}] ts_first: {head.text!r} {head.start_time}->{head.end_time} s")
            print(f"[sample {i}] ts_last : {tail.text!r} {tail.start_time}->{tail.end_time} s")


