# Qwen3-TTS Playground

Qwen3-TTS を試すためのプレイグラウンド。バックエンド (FastAPI) とフロントエンド (React + Vite) 構成。
環境構築の詳細は [docs/step01-environment-setup.md](docs/step01-environment-setup.md) を参照。

## 起動

ターミナルを2つ使う。

```bash
# 1. バックエンド (初回はモデルのダウンロードと読み込みで数分かかる)
cd backend
uv run uvicorn qwen3_tts_spike.server:app --port 8765

# 2. フロントエンド
cd frontend
npm install   # 初回のみ
npm run dev
```

ブラウザで <http://localhost:5173> を開く。`/api` は Vite のプロキシで `127.0.0.1:8765` に中継される。

## 使い方

テキスト・言語・話者を選んで「生成」を押す。生成した音声は履歴に積まれ、各行に音声長・生成時間・RTF が出る。

## 構成

| パス | 内容 |
| --- | --- |
| `backend/src/qwen3_tts_spike/server.py` | API サーバ。起動時にモデルを1回だけロードし、GPU は1リクエストずつ処理する |
| `frontend/src/api.ts` | API クライアント |
| `frontend/src/App.tsx` | UI |

### API

| メソッド | パス | 内容 |
| --- | --- | --- |
| GET | `/api/info` | モデル名・話者一覧・言語一覧・instruct 対応可否 |
| GET | `/api/models` | 選択可能なモデル、必要 VRAM の目安、現在の空き VRAM、収まるか (`fits`) |
| POST | `/api/models/select` | `{model}` でモデルを切り替える。現在のモデルを解放した後の空きで判定し、足りなければ 409 を返して元のモデルに戻す |
| POST | `/api/tts` | `{text, language, speaker, instruct}` → `audio/wav`。生成時間などはレスポンスヘッダ `X-Generate-Seconds` / `X-Audio-Seconds` |

## モデルの切り替え

画面上部のモデル選択で切り替える。空き VRAM に収まらないモデルは選択できない (「空き不足」と表示)。
切り替え時は現在のモデルを解放してから読み込むため、数十秒〜数分かかり、その後の最初の生成もウォームアップで遅い。
起動時のモデルは環境変数 `QWEN_TTS_MODEL` で指定できる (既定は `Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice`)。

| モデル | 必要 VRAM の目安 | instruct |
| --- | --- | --- |
| 0.6B CustomVoice | 2.5GiB (実測 2.18GiB + 余裕) | 非対応 |
| 1.7B CustomVoice | 5.0GiB (実測: 空き 5.00GiB を使い切って動作) | 対応 |

実測は `uv run python scripts/measure_vram.py <model_id>` で再現できる (空き 5.00GiB の RTX 3060 Laptop、bf16、sdpa)。

| モデル | 読み込み | 確保 VRAM (allocated / reserved) | 生成 (1回目 → 2回目 RTF) |
| --- | --- | --- | --- |
| 0.6B | 11.0s | 2.01 / 2.13GiB | 4.94 → 2.37 |
| 1.7B | 135.3s (初回ダウンロード込み) | 3.89 / 3.90GiB (ピーク reserved 4.40GiB) | 7.26 → 2.14 |

1.7B は CUDA コンテキスト等を含めて 5.0GiB ちょうどまで使い、空きが 0 になった。余裕が無いので、
他アプリの使用量次第で OOM になりうる。`required_gib` は [server.py](backend/src/qwen3_tts_spike/server.py) の `MODELS` で調整できる。
空き VRAM は他アプリの使用分を除いて判定するため、この開発機では空きが 4.99GiB になると 1.7B は選べない (閾値 5.0 との差は 0.01GiB)。

## 制限事項

- 現在対応しているのは CustomVoice 系のモデルのみ (VoiceDesign / Base によるクローンは未対応)。
- 1回の生成が終わるまで音声は返らない (ストリーミングなし)。
- テキストは 300 文字まで。
- 8000 番ポートは環境によっては他のプロセスが使っているため、バックエンドは 8765 にしている。
