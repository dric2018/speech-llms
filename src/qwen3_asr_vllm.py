# coding=utf-8
# Copyright 2026 The Alibaba Qwen team.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from config import CFG

import torch

from qwen_asr import Qwen3ASRModel

import logging

from utils import (_print_result, 
                   _download_audio_bytes, 
                   _read_wav_from_bytes, 
                   _to_data_url_base64,
                )

logging.basicConfig(level="INFO")    

def test_single_url(
        asr: Qwen3ASRModel
    ) -> None:

    logging.info("Testing single URL..")

    results = asr.transcribe(
        audio=CFG.URL_ZH,
        language=None,
        return_time_stamps=False,
    )
    assert isinstance(results, list) and len(results) == 1
    _print_result("single-url (no forced language, no timestamps)", results)


def test_batch_mixed(
    asr: Qwen3ASRModel
    ) -> None:
    logging.info("Testing batched with mixed inputs URL...")

    zh_bytes = _download_audio_bytes(CFG.URL_ZH)
    en_bytes = _download_audio_bytes(CFG.URL_EN)

    zh_b64 = _to_data_url_base64(zh_bytes, mime="audio/wav")
    en_wav, en_sr = _read_wav_from_bytes(en_bytes)

    logging.info("Transcribing...")

    results = asr.transcribe(
        audio=[CFG.URL_ZH, zh_b64, (en_wav, en_sr)],
        context=["", "交易 停滞", ""],
        language=[None, "Chinese", "English"],
        return_time_stamps=False,
    )
    assert len(results) == 3
    _print_result("batch-mixed (forced language for some)", results)


def test_single_with_timestamps(asr: Qwen3ASRModel) -> None:
    results = asr.transcribe(
        audio=CFG.URL_EN,
        language="English",
        return_time_stamps=True,
    )
    assert len(results) == 1
    assert results[0].time_stamps is not None
    _print_result("single-url (forced language + timestamps)", results)


def test_batch_with_timestamps(asr: Qwen3ASRModel) -> None:
    zh_bytes = _download_audio_bytes(CFG.URL_ZH)
    zh_b64 = _to_data_url_base64(zh_bytes, mime="audio/wav")

    results = asr.transcribe(
        audio=[CFG.URL_ZH, zh_b64, CFG.URL_EN],
        context=["", "交易 停滞", ""],
        language=["Chinese", "Chinese", "English"],
        return_time_stamps=True,
    )
    assert len(results) == 3
    assert all(r.time_stamps is not None for r in results)
    _print_result("batch (forced language + timestamps)", results)


def main() -> None:
    logging.info("Building ASR model...")
    asr = Qwen3ASRModel.LLM(
        model=CFG.ASR_MODEL_PATH,
        gpu_memory_utilization=CFG.GPU_MEM_PCT,
        forced_aligner=CFG.FORCED_ALIGNER_PATH,
        forced_aligner_kwargs=dict(
            dtype=torch.bfloat16,
            device_map=CFG.DEVICE,
            # attn_implementation="flash_attention_2",
        ),
        max_inference_batch_size=CFG.BATCH_SIZE,
        max_new_tokens=CFG.MAX_NEW_TOKENS,
    )

    logging.info("Done building ASR model.")

    test_single_url(asr)
    test_batch_mixed(asr)
    # test_single_with_timestamps(asr)
    # test_batch_with_timestamps(asr)


if __name__ == "__main__":
    main()