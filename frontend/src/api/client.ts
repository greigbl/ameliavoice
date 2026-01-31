import axios from 'axios'

const api = axios.create({
  baseURL: '/api',
  headers: { 'Content-Type': 'application/json' },
  timeout: 60000,
})

export interface TurnLatency {
  stt_ms: number
  llm_ms: number
  tts_ms: number
}

export interface ChatMessage {
  role: 'user' | 'assistant' | 'system'
  content: string
  /** STT/LLM/TTS durations (ms) for voice turns; only set on assistant messages from voice pipeline */
  latency?: TurnLatency
}

export interface TranscribeResponse {
  text: string
  language: string
  model: string
}

export interface ChatResponse {
  message: ChatMessage
  done: boolean
  /** When true, agent invoked end_conversation; client should stop listening and call POST /api/voice/end */
  end_conversation?: boolean
}

export async function transcribe(
  audioBlob: Blob,
  language: string,
  model: string
): Promise<TranscribeResponse> {
  const form = new FormData()
  const ext = audioBlob.type.includes('webm') ? 'webm' : 'wav'
  form.append('audio', audioBlob, `audio.${ext}`)
  form.append('language', language === 'JA' ? 'ja' : 'en')
  form.append('model', model)
  const { data } = await api.post<TranscribeResponse>('/transcribe', form, {
    headers: { 'Content-Type': 'multipart/form-data' },
  })
  return data
}

export async function textToSpeech(text: string, languageCode: string): Promise<Blob> {
  const { data } = await api.post<Blob>(
    '/tts',
    { text, language_code: languageCode },
    { responseType: 'blob' }
  )
  return data
}

export async function chat(
  messages: ChatMessage[],
  integration: string,
  language: string = 'en',
  verbosity?: 'brief' | 'normal' | 'detailed'
): Promise<ChatResponse> {
  const body: {
    messages: { role: string; content: string }[]
    integration: string
    language: string
    verbosity?: string
  } = {
    messages: messages.map((m) => ({ role: m.role, content: m.content })),
    integration,
    language: language === 'JA' ? 'ja' : 'en',
  }
  if (verbosity != null) body.verbosity = verbosity
  const { data } = await api.post<ChatResponse>('/chat', body)
  return data
}

export async function health(): Promise<{ ok: boolean; stt_available: boolean; tts_available: boolean }> {
  const { data } = await api.get('/health')
  return data
}

/** Signal that the voice session should end (close listening). Call when agent returns end_conversation. */
export async function endVoiceSession(): Promise<{ ok: boolean }> {
  const { data } = await api.post<{ ok: boolean }>('/voice/end')
  return data
}

// Voice calls (Twilio) - list and detail with transcript + latency
export interface VoiceCallSummary {
  call_sid: string
  stream_sid: string
  start_time: number | null
  end_time: number | null
  turn_count: number
}

export interface VoiceCallTurn {
  user_text: string
  assistant_text: string
  stt_ms: number
  llm_ms: number
  tts_ms: number
}

export interface VoiceCallDetail {
  call_sid: string
  stream_sid: string
  start_time: number | null
  end_time: number | null
  turns: VoiceCallTurn[]
}

export async function getVoiceCalls(): Promise<VoiceCallSummary[]> {
  const { data } = await api.get<VoiceCallSummary[]>('/voice/calls')
  return data
}

export async function getVoiceCall(callSid: string): Promise<VoiceCallDetail> {
  const { data } = await api.get<VoiceCallDetail>(`/voice/calls/${callSid}`)
  return data
}
