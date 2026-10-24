# Qwen3-TTS を uv + React 構成で動かすまで（WSL2 / RTX 3060 Laptop 6GB）

Alibaba Qwen チームの音声合成モデル [Qwen3-TTS](https://github.com/QwenLM/Qwen3-TTS) を、**Python は uv、フロントエンドは React（Vite）** という構成で動かし、日本語の音声を1本生成するまでの手順をまとめます。実際に詰まった点とその解決策も含めています。

## この記事のゴール

- uv で Python 3.12 環境を作り、`qwen-tts` を導入する
- GPU（CUDA）上で 0.6B モデルを動かし、日本語の wav を生成する
- React（Vite + TypeScript）のフロントエンドをビルドできる状態にする

## 検証環境

| 項目 | 内容 |
| --- | --- |
| OS | Windows 11 + WSL2 (Ubuntu 24.04, Linux 6.18 WSL2 kernel) |
| GPU | NVIDIA GeForce RTX 3060 Laptop GPU（VRAM 6GB、検証時は他アプリが約2GB使用） |
| RAM | 約 7GB（WSL） |
| ドライバ | NVIDIA 610.47（CUDA 13.3 対応） |
| uv | 0.13.5 |
| Node.js | v24（nvm で導入済み） |
| 作業場所 | `/mnt/c/...`（Windows側 NTFS 上） |

前提として、WSL2 で `nvidia-smi` が GPU を表示できる状態にしてください。CUDA Toolkit を WSL に別途入れる必要はありません（PyTorch の wheel が CUDA ランタイムを同梱しています）。

## 結論（先に手順だけ）

```bash
# 1. Python 側
mkdir backend && cd backend
uv init --no-readme --python 3.12 --name qwen3-tts-spike
# pyproject.toml の末尾に [tool.uv] 設定を追記（後述）
uv add qwen-tts

# 2. 動作確認
uv run python scripts/hello_tts.py

# 3. フロントエンド側
cd ..
npm create vite@latest frontend -- --template react-ts --no-interactive
cd frontend && npm install && npm run build
```

ただし **そのまま実行すると Triton のコンパイルエラーで失敗します**。次の「ハマりどころ」を先に読んでください。

## ハマりどころ（重要）

### 1. システムの Python を使うと Triton が `Python.h` が無くて落ちる

uv は、`requires-python` を満たすシステムの Python（`/usr/bin/python3.12`）があるとそれを優先して使います。Ubuntu の標準 Python には開発用ヘッダ（`python3-dev`）が入っていないため、モデル実行時に Triton が C 拡張を JIT コンパイルする段階で失敗します。

```
fatal error: Python.h: そのようなファイルやディレクトリはありません
subprocess.CalledProcessError: Command '['/usr/bin/gcc', '.../triton/backends/nvidia/driver.c', ...
```

対処は 2 通りあります。

- `sudo apt install python3.12-dev` を入れる
- **uv が管理する Python（ヘッダ同梱）を使う** ← 今回はこちら（sudo 不要）

`backend/pyproject.toml` の末尾に以下を追記します。

```toml
[tool.uv]
python-preference = "only-managed"   # uv管理のPythonだけを使う
link-mode = "copy"                   # 次項参照
```

すでに `.venv` を作ってしまった場合は、削除して作り直してください。

```bash
rm -rf .venv
uv python install 3.12
uv sync
```

作り直し後、`Python.h` が存在することを確認できます。

```bash
.venv/bin/python -c "import sysconfig,os;print(os.path.exists(sysconfig.get_paths()['include']+'/Python.h'))"
# True
```

### 2. NTFS（/mnt/c）上に作ると非常に遅い

プロジェクトを `/mnt/c/...` に置くと、`.venv` の書き込み・読み込みが WSL の 9P 経由になり、かなり遅くなります。今回の実測は次のとおりです。

| 操作 | 所要時間 |
| --- | --- |
| `uv add qwen-tts`（torch 等 約 110 パッケージ） | 約 8 分 30 秒 |
| `.venv` 作り直し（`uv sync`、キャッシュあり） | 約 8 分 |
| 初回の `import torch` / `import qwen_tts` | 約 3 分 |

また、uv のキャッシュ（`~/.cache/uv`、ext4）とプロジェクトが別ファイルシステムなので、既定のハードリンクが使えず警告が出ます。`link-mode = "copy"` を設定して警告を避けています。

**時間を気にするなら、プロジェクトを WSL の Linux 側（`~/projects/...`）に置くのがおすすめです。** 今回は Windows 側のエディタとの併用を優先して NTFS のまま進めました。

### 3. 起動時の警告（無視してよいもの）

- `SoX could not be found!` … `sox` コマンドが無いという警告です。今回の生成には影響しませんでした。気になる場合は `sudo apt install sox` を実行してください。
- `flash-attn is not installed` … 次節参照。

## 手順の詳細

### 1. Python 環境の構築

```bash
mkdir backend && cd backend
uv init --no-readme --python 3.12 --name qwen3-tts-spike
```

上記の「ハマりどころ 1」の `[tool.uv]` を `pyproject.toml` に追記してから、パッケージを追加します。

```bash
uv add qwen-tts
```

公式 README は conda + `pip install -U qwen-tts` を案内していますが、uv なら `uv add` で同じパッケージを入れられます。主な解決結果は次のとおりです。

| パッケージ | バージョン |
| --- | --- |
| qwen-tts | 0.1.1 |
| torch / torchaudio | 2.14.1+cu130 / 2.11.0+cu130 |
| transformers | 4.57.3（qwen-tts が固定） |
| accelerate | 1.12.0（同上） |
| triton | 3.8.0 |

PyTorch は既定で CUDA 13.0 ビルドが入り、そのまま RTX 3060 で使えました。確認は次のとおりです。

```bash
uv run python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))"
# 2.14.1+cu130 True NVIDIA GeForce RTX 3060 Laptop GPU
```

#### FlashAttention 2 は入れない

公式は VRAM 削減のために `flash-attn` を推奨していますが、ソースビルドが重く、RAM 7GB の環境では現実的ではないため入れていません。代わりにモデル読み込み時に `attn_implementation="sdpa"` を指定します（未導入だと警告が出ますが動作します）。

### 2. モデルの選択

README のモデル一覧のうち、VRAM 6GB（実質空き約 4GB）で動かすため **`Qwen3-TTS-12Hz-0.6B-CustomVoice`** を選びました。

| モデル | 用途 |
| --- | --- |
| 1.7B / 0.6B-CustomVoice | 用意された 9 種の話者から選んで読み上げる（`instruct` で口調指定） |
| 1.7B-VoiceDesign | 文章で声質を設計する |
| 1.7B / 0.6B-Base | 3 秒の参照音声から声をクローンする |

モデルの重みは `from_pretrained` の初回呼び出し時に Hugging Face から自動ダウンロードされます（手動ダウンロードは不要）。

### 3. 日本語音声を生成する

`backend/scripts/hello_tts.py`:

```python
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
    speaker="Ono_Anna",   # 日本語話者
)
elapsed = time.perf_counter() - t0
duration = len(wavs[0]) / sr
print(f"generate: {elapsed:.1f}s for {duration:.1f}s audio (RTF={elapsed / duration:.2f})")
print(f"peak VRAM: {torch.cuda.max_memory_allocated() / 1024**3:.2f} GiB")

sf.write("output_hello.wav", wavs[0], sr)
```

実行:

```bash
uv run python scripts/hello_tts.py
```

実際の出力（抜粋）:

```
load: 24.7s
speakers: ['aiden', 'dylan', 'eric', 'ono_anna', 'ryan', 'serena', 'sohee', 'uncle_fu', 'vivian']
languages: ['auto', 'chinese', 'english', 'french', 'german', 'italian', 'japanese', 'korean', 'portuguese', 'russian', 'spanish']
generate: 49.7s for 7.5s audio (RTF=6.61)
peak VRAM: 2.23 GiB
saved output_hello.wav
```

`output_hello.wav`（7.5 秒）が生成されれば成功です。

> 補足: `get_supported_speakers()` は小文字で返りますが、README では `Ono_Anna` と大文字小文字混じりで書かれています。今回は README の表記で動作しました。

### 4. 計測結果と注意点

| 指標 | 値 |
| --- | --- |
| モデル読み込み | 24.7 秒（ダウンロード済み・NTFS 上） |
| 生成時間 | 49.7 秒（出力音声 7.5 秒） |
| RTF（生成時間 ÷ 音声長） | 6.61 |
| ピーク VRAM | 2.23 GiB |

**RTF が 1 を大きく超えており、この構成ではリアルタイム再生はできません。** 計測は 1 回のみで、ウォームアップ（初回の CUDA カーネル準備）を含みます。また GPU は他アプリと共有しており、flash-attn も未使用です。ストリーミング・低遅延化（本プロジェクト名の "rt"）は、2 回目以降の実測や最適化を含め、後続ステップで検証します。

### 5. React フロントエンドの構築

```bash
cd ..   # プロジェクトルート
npm create vite@latest frontend -- --template react-ts --no-interactive
cd frontend
npm install
npm run build
```

`npm run build` が `dist/` を出力して成功すれば完了です（今回は約 1 秒）。開発サーバは `npm run dev` で起動できます。バックエンドとの接続は後続ステップで行います。

## 最終的なディレクトリ構成

```
06_qwen3-tts-rt/
├── backend/
│   ├── pyproject.toml        # qwen-tts 依存 + [tool.uv] 設定
│   ├── uv.lock
│   ├── .python-version       # 3.12
│   └── scripts/hello_tts.py  # 動作確認スクリプト
├── frontend/                 # Vite + React + TypeScript
└── docs/
    ├── spec.md
    └── step01-environment-setup.md  # この記事
```

## まとめ

- uv だけで Qwen3-TTS の環境は作れる。公式の conda 手順は必須ではない。
- **uv 管理の Python を使うこと**（`python-preference = "only-managed"`）。システム Python だと Triton が `Python.h` 不足で失敗する。
- 6GB VRAM でも 0.6B-CustomVoice は動く（ピーク 2.23GiB）。ただし今回の構成での RTF は 6.61 で、リアルタイム用途には最適化が必要。
- NTFS 上のプロジェクトは導入・初回 import が非常に遅い。可能なら WSL 側のファイルシステムに置く。
