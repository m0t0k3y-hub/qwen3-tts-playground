export interface Info {
  model: string
  speakers: string[]
  languages: string[]
  supports_instruct: boolean
  max_text_length: number
  load_seconds: number
}

export interface TtsParams {
  text: string
  language: string
  speaker: string
  instruct: string
}

export interface TtsResult {
  url: string
  generateSeconds: number
  audioSeconds: number
}

export async function fetchInfo(): Promise<Info> {
  const res = await fetch('/api/info')
  if (!res.ok) throw new Error(`info: HTTP ${res.status}`)
  return res.json()
}

export interface ModelOption {
  id: string
  label: string
  required_gib: number
  estimated: boolean
  supports_instruct: boolean
  fits: boolean
}

export interface Models {
  current: string
  total_vram_gib: number
  available_vram_gib: number
  models: ModelOption[]
}

async function errorMessage(res: Response): Promise<string> {
  const body = await res.json().catch(() => null)
  const detail = body?.detail
  if (!detail) return `HTTP ${res.status}`
  return typeof detail === 'string' ? detail : JSON.stringify(detail)
}

export async function fetchModels(): Promise<Models> {
  const res = await fetch('/api/models')
  if (!res.ok) throw new Error(await errorMessage(res))
  return res.json()
}

export async function selectModel(model: string): Promise<Models> {
  const res = await fetch('/api/models/select', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ model }),
  })
  if (!res.ok) throw new Error(await errorMessage(res))
  return res.json()
}

export async function synthesize(params: TtsParams, signal?: AbortSignal): Promise<TtsResult> {
  const res = await fetch('/api/tts', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(params),
    signal,
  })
  if (!res.ok) throw new Error(await errorMessage(res))
  const blob = await res.blob()
  return {
    url: URL.createObjectURL(blob),
    generateSeconds: Number(res.headers.get('X-Generate-Seconds')),
    audioSeconds: Number(res.headers.get('X-Audio-Seconds')),
  }
}
