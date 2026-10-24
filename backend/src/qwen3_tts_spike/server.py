"""Qwen3-TTS プレイグラウンド用 API サーバ。

起動: uv run uvicorn qwen3_tts_spike.server:app --port 8765
起動時のモデルは QWEN_TTS_MODEL 環境変数で指定する (既定: 0.6B-CustomVoice)。
実行中は /api/models/select で VRAM の空きに収まるモデルへ切り替えられる。
"""
import gc
import io
import os
import threading
import time
from contextlib import asynccontextmanager

import soundfile as sf
import torch
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response
from pydantic import BaseModel, Field
from qwen_tts import Qwen3TTSModel

GIB = 1024**3
MAX_TEXT_LENGTH = 300

# 選択可能なモデル。generate_custom_voice で扱える CustomVoice 系のみ。
# required_gib: 読み込みに必要な VRAM の目安。
#   scripts/measure_vram.py での実測 (RTX 3060 Laptop, bf16, sdpa)。CUDA コンテキスト等を含め、
#   読み込み〜生成後に空き VRAM が減った量: 0.6B = 2.18GiB (閾値は余裕を足して 2.5)、
#   1.7B = 5.00GiB (空き 5.00GiB から 0.00GiB まで使い切った。余裕はほぼ無い)。
MODELS = {
    "Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice": {
        "label": "0.6B CustomVoice",
        "required_gib": 2.5,
        "estimated": False,
        "supports_instruct": False,  # README の対応表では instruct は 1.7B のみ
    },
    "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice": {
        "label": "1.7B CustomVoice",
        "required_gib": 5.0,
        "estimated": False,
        "supports_instruct": True,
    },
}
DEFAULT_MODEL = "Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice"

state: dict = {"model": None, "model_id": None, "load_seconds": 0.0}
# GPU は同時に1リクエストしか処理しない (生成中の切り替えも防ぐ)
gpu_lock = threading.Lock()


def available_vram_gib() -> float:
    """新しいモデルに使える VRAM。他プロセスの使用分は除き、自分が確保済みの分は含める。"""
    free, _ = torch.cuda.mem_get_info()
    return (free + torch.cuda.memory_reserved()) / GIB


def unload() -> None:
    state["model"] = None
    state["model_id"] = None
    gc.collect()
    torch.cuda.empty_cache()


def load(model_id: str) -> None:
    unload()
    t0 = time.perf_counter()
    state["model"] = Qwen3TTSModel.from_pretrained(
        model_id,
        device_map="cuda:0",
        dtype=torch.bfloat16,
        attn_implementation="sdpa",
    )
    state["model_id"] = model_id
    state["load_seconds"] = time.perf_counter() - t0


@asynccontextmanager
async def lifespan(_: FastAPI):
    model_id = os.environ.get("QWEN_TTS_MODEL", DEFAULT_MODEL)
    if model_id not in MODELS:
        raise RuntimeError(f"QWEN_TTS_MODEL must be one of {list(MODELS)}")
    load(model_id)
    yield
    unload()


app = FastAPI(title="Qwen3-TTS Playground", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["X-Generate-Seconds", "X-Audio-Seconds", "X-Sample-Rate"],
)


class TtsRequest(BaseModel):
    text: str = Field(min_length=1, max_length=MAX_TEXT_LENGTH)
    language: str = "auto"
    speaker: str
    instruct: str = ""


class SelectRequest(BaseModel):
    model: str


def models_payload() -> dict:
    available = available_vram_gib()
    total = torch.cuda.get_device_properties(0).total_memory / GIB
    return {
        "current": state["model_id"],
        "total_vram_gib": round(total, 2),
        "available_vram_gib": round(available, 2),
        "models": [
            {
                "id": model_id,
                "label": meta["label"],
                "required_gib": meta["required_gib"],
                "estimated": meta["estimated"],
                "supports_instruct": meta["supports_instruct"],
                "fits": available >= meta["required_gib"],
            }
            for model_id, meta in MODELS.items()
        ],
    }


@app.get("/api/info")
def info():
    model = state["model"]
    meta = MODELS[state["model_id"]]
    return {
        "model": state["model_id"],
        "speakers": model.get_supported_speakers(),
        "languages": model.get_supported_languages(),
        "supports_instruct": meta["supports_instruct"],
        "max_text_length": MAX_TEXT_LENGTH,
        "load_seconds": state["load_seconds"],
    }


@app.get("/api/models")
def list_models():
    return models_payload()


@app.post("/api/models/select")
def select_model(req: SelectRequest):
    meta = MODELS.get(req.model)
    if meta is None:
        raise HTTPException(400, f"unknown model: {req.model}")
    with gpu_lock:
        if req.model == state["model_id"]:
            return models_payload()
        previous = state["model_id"]
        # 解放した後の空きで判定する (現在のモデルの分は戻ってくる)
        unload()
        available = available_vram_gib()
        if available < meta["required_gib"]:
            load(previous)
            raise HTTPException(
                409,
                f"VRAM が足りません: 空き {available:.2f}GiB / 必要 {meta['required_gib']:.2f}GiB",
            )
        try:
            load(req.model)
        except torch.cuda.OutOfMemoryError as e:
            load(previous)
            raise HTTPException(409, f"読み込み中に VRAM が不足しました: {e}") from e
    return models_payload()


@app.post("/api/tts")
def tts(req: TtsRequest):
    # 同期関数なので FastAPI のスレッドプールで実行される
    with gpu_lock:
        model = state["model"]
        supports_instruct = MODELS[state["model_id"]]["supports_instruct"]
        if req.speaker not in model.get_supported_speakers():
            raise HTTPException(400, f"unknown speaker: {req.speaker}")
        if req.language not in model.get_supported_languages():
            raise HTTPException(400, f"unknown language: {req.language}")

        kwargs = {}
        if supports_instruct and req.instruct.strip():
            kwargs["instruct"] = req.instruct.strip()

        t0 = time.perf_counter()
        wavs, sr = model.generate_custom_voice(
            text=req.text,
            language=req.language,
            speaker=req.speaker,
            **kwargs,
        )
        elapsed = time.perf_counter() - t0

    buf = io.BytesIO()
    sf.write(buf, wavs[0], sr, format="WAV")
    return Response(
        buf.getvalue(),
        media_type="audio/wav",
        headers={
            "X-Generate-Seconds": f"{elapsed:.2f}",
            "X-Audio-Seconds": f"{len(wavs[0]) / sr:.2f}",
            "X-Sample-Rate": str(sr),
        },
    )
