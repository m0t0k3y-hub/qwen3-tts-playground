import { useEffect, useRef, useState } from 'react'
import './App.css'
import {
  fetchInfo,
  fetchModels,
  selectModel,
  synthesize,
  type Info,
  type Models,
  type TtsParams,
  type TtsResult,
} from './api'

interface HistoryItem extends TtsParams, TtsResult {
  id: number
}

const SAMPLES = [
  { label: '日本語', language: 'japanese', speaker: 'ono_anna', text: 'こんにちは。これはQwen3-TTSの動作確認です。' },
  { label: 'English', language: 'english', speaker: 'ryan', text: 'She said she would be here by noon.' },
  { label: '中文', language: 'chinese', speaker: 'vivian', text: '其实我真的有发现，我是一个特别善于观察别人情绪的人。' },
]

export default function App() {
  const [info, setInfo] = useState<Info | null>(null)
  const [error, setError] = useState('')
  const [params, setParams] = useState<TtsParams>({
    text: SAMPLES[0].text,
    language: SAMPLES[0].language,
    speaker: SAMPLES[0].speaker,
    instruct: '',
  })
  const [models, setModels] = useState<Models | null>(null)
  const [switching, setSwitching] = useState(false)
  const [busy, setBusy] = useState(false)
  const [elapsed, setElapsed] = useState(0)
  const [history, setHistory] = useState<HistoryItem[]>([])
  const nextId = useRef(1)

  useEffect(() => {
    Promise.all([fetchInfo(), fetchModels()])
      .then(([i, m]) => {
        setInfo(i)
        setModels(m)
      })
      .catch((e) => setError(`バックエンドに接続できません: ${e.message}`))
  }, [])

  async function refreshModels() {
    try {
      setModels(await fetchModels())
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e))
    }
  }

  async function changeModel(id: string) {
    setSwitching(true)
    setError('')
    try {
      setModels(await selectModel(id))
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e))
    } finally {
      // 切り替えの成否にかかわらず、実際にロードされているモデルの状態を取り直す
      try {
        const [i, m] = await Promise.all([fetchInfo(), fetchModels()])
        setInfo(i)
        setModels(m)
        // 話者・言語がモデルで変わりうるので、使えない値は先頭に寄せる
        setParams((p) => ({
          ...p,
          speaker: i.speakers.includes(p.speaker) ? p.speaker : i.speakers[0],
          language: i.languages.includes(p.language) ? p.language : i.languages[0],
          instruct: i.supports_instruct ? p.instruct : '',
        }))
      } catch (e) {
        setError(e instanceof Error ? e.message : String(e))
      }
      setSwitching(false)
    }
  }

  // 生成は数十秒かかるので経過時間を表示する
  useEffect(() => {
    if (!busy) return
    const start = Date.now()
    const timer = setInterval(() => setElapsed((Date.now() - start) / 1000), 200)
    return () => clearInterval(timer)
  }, [busy])

  const set = <K extends keyof TtsParams>(key: K, value: TtsParams[K]) =>
    setParams((p) => ({ ...p, [key]: value }))

  async function generate() {
    setElapsed(0)
    setBusy(true)
    setError('')
    try {
      const result = await synthesize(params)
      setHistory((h) => [{ ...params, ...result, id: nextId.current++ }, ...h])
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e))
    } finally {
      setBusy(false)
    }
  }

  const maxLen = info?.max_text_length ?? 300

  return (
    <main className="playground">
      <h1>Qwen3-TTS Playground</h1>
      {error && <p className="error" role="alert">{error}</p>}

      <section className="card">
        <label>
          モデル
          <select
            value={models?.current ?? ''}
            onChange={(e) => changeModel(e.target.value)}
            disabled={!models || switching || busy}
          >
            {models?.models.map((m) => (
              <option key={m.id} value={m.id} disabled={!m.fits && m.id !== models.current}>
                {m.label}（VRAM {m.estimated ? '約' : ''}{m.required_gib}GiB 必要）
                {!m.fits && m.id !== models.current ? ' — 空き不足' : ''}
              </option>
            ))}
          </select>
        </label>
        {models && (
          <p className="meta">
            VRAM 空き {models.available_vram_gib.toFixed(1)} / {models.total_vram_gib.toFixed(1)} GiB
            （現在のモデルの分を含む）
            {info && ` · 読み込み ${info.load_seconds.toFixed(1)}s`}
            {' '}
            <button type="button" className="link" onClick={refreshModels} disabled={switching}>
              再取得
            </button>
          </p>
        )}
        {switching && <p className="meta">モデルを切り替えています…（読み込みに数十秒〜数分かかります）</p>}
      </section>

      <section className="card">
        <div className="samples">
          {SAMPLES.map((s) => (
            <button
              key={s.label}
              type="button"
              className="chip"
              onClick={() => setParams((p) => ({ ...p, text: s.text, language: s.language, speaker: s.speaker }))}
            >
              {s.label}
            </button>
          ))}
        </div>

        <label>
          テキスト
          <textarea
            rows={4}
            maxLength={maxLen}
            value={params.text}
            onChange={(e) => set('text', e.target.value)}
          />
          <span className="count">{params.text.length} / {maxLen}</span>
        </label>

        <div className="row">
          <label>
            言語
            <select value={params.language} onChange={(e) => set('language', e.target.value)} disabled={!info}>
              {info?.languages.map((l) => <option key={l}>{l}</option>)}
            </select>
          </label>
          <label>
            話者
            <select value={params.speaker} onChange={(e) => set('speaker', e.target.value)} disabled={!info}>
              {info?.speakers.map((s) => <option key={s}>{s}</option>)}
            </select>
          </label>
        </div>

        <label>
          instruct（口調の指示）
          <input
            value={params.instruct}
            onChange={(e) => set('instruct', e.target.value)}
            disabled={!info?.supports_instruct}
            placeholder={info?.supports_instruct ? '例: Very happy.' : 'このモデルは instruct 非対応'}
          />
        </label>

        <button type="button" className="primary" onClick={generate} disabled={busy || switching || !info || !params.text.trim()}>
          {busy ? `生成中… ${elapsed.toFixed(1)}s` : '生成'}
        </button>
      </section>

      <section className="history-section">
        <h2>履歴</h2>
        {history.length === 0 && <p className="meta">まだ生成していません。</p>}
        <ul className="history">
          {history.map((h) => (
            <li key={h.id} className="card">
              <p>{h.text}</p>
              <p className="meta">
                {h.language} / {h.speaker} · 音声 {h.audioSeconds.toFixed(1)}s · 生成 {h.generateSeconds.toFixed(1)}s ·
                RTF {(h.generateSeconds / h.audioSeconds).toFixed(2)}
              </p>
              <audio controls src={h.url} autoPlay={h.id === history[0].id} />
            </li>
          ))}
        </ul>
      </section>
    </main>
  )
}
