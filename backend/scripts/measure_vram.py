"""モデルの VRAM 使用量を実測する。使い方: uv run python scripts/measure_vram.py <model_id>"""
import sys
import time

import torch
from qwen_tts import Qwen3TTSModel

GIB = 1024**3
model_id = sys.argv[1]
free0, total = torch.cuda.mem_get_info()
print(f"before: free {free0 / GIB:.2f} / total {total / GIB:.2f} GiB")

t0 = time.perf_counter()
try:
    model = Qwen3TTSModel.from_pretrained(
        model_id, device_map="cuda:0", dtype=torch.bfloat16, attn_implementation="sdpa"
    )
except torch.cuda.OutOfMemoryError as e:
    print(f"OOM at load: {str(e)[:200]}")
    print(f"peak allocated: {torch.cuda.max_memory_allocated() / GIB:.2f} GiB")
    sys.exit(2)
print(f"load: {time.perf_counter() - t0:.1f}s")
print(f"after load: allocated {torch.cuda.memory_allocated() / GIB:.2f}, reserved {torch.cuda.memory_reserved() / GIB:.2f} GiB")

kwargs = {"instruct": "Very happy."} if "1.7B" in model_id else {}
for i in range(2):
    t0 = time.perf_counter()
    try:
        wavs, sr = model.generate_custom_voice(
            text="こんにちは。テストです。", language="japanese", speaker="ono_anna", **kwargs
        )
    except torch.cuda.OutOfMemoryError as e:
        print(f"OOM at generate #{i + 1}: {str(e)[:200]}")
        print(f"peak allocated: {torch.cuda.max_memory_allocated() / GIB:.2f} GiB")
        sys.exit(3)
    el = time.perf_counter() - t0
    dur = len(wavs[0]) / sr
    print(f"generate #{i + 1}: {el:.1f}s for {dur:.1f}s (RTF {el / dur:.2f})")

print(f"peak allocated: {torch.cuda.max_memory_allocated() / GIB:.2f} GiB, reserved: {torch.cuda.memory_reserved() / GIB:.2f} GiB")
free1, _ = torch.cuda.mem_get_info()
print(f"after: free {free1 / GIB:.2f} GiB")
