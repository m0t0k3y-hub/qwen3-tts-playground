"""Qwen3-TTS 動作確認: CustomVoice (0.6B) で日本語音声を1本生成する。"""
import time

import soundfile as sf
import torch
from qwen_tts import Qwen3TTSModel

MODEL_ID = "Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice"

t0 = time.perf_counter()
model = Qwen3TTSModel.from_pretrained(
    MODEL_ID,
    device_map="cuda:0",
    dtype=torch.bfloat16,
    attn_implementation="sdpa",  # flash-attn なしで動かす
)
print(f"load: {time.perf_counter() - t0:.1f}s")
print("speakers:", model.get_supported_speakers())
print("languages:", model.get_supported_languages())

t0 = time.perf_counter()
wavs, sr = model.generate_custom_voice(
    text="こんにちは。これはQwen3-TTSの動作確認です。",
    language="Japanese",
    speaker="Ono_Anna",
)
elapsed = time.perf_counter() - t0
duration = len(wavs[0]) / sr
print(f"generate: {elapsed:.1f}s for {duration:.1f}s audio (RTF={elapsed / duration:.2f})")
print(f"peak VRAM: {torch.cuda.max_memory_allocated() / 1024**3:.2f} GiB")

sf.write("output_hello.wav", wavs[0], sr)
print("saved output_hello.wav")
