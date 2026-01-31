import { useEffect, useRef, useState } from 'react'

export interface LiveEvent {
  call_sid: string
  event: 'stt_done' | 'llm_done' | 'tts_start' | 'tts_done'
  payload: Record<string, unknown>
}

function getWsUrl(path: string): string {
  const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:'
  const host = window.location.host
  return `${protocol}//${host}${path}`
}

export function useVoiceCallLive(callSid: string | undefined, onMessage: (ev: LiveEvent) => void) {
  const [connected, setConnected] = useState(false)
  const wsRef = useRef<WebSocket | null>(null)
  const onMessageRef = useRef(onMessage)
  onMessageRef.current = onMessage

  useEffect(() => {
    if (!callSid) return
    const url = getWsUrl('/api/voice/calls/live')
    const ws = new WebSocket(url)
    wsRef.current = ws

    ws.onopen = () => {
      ws.send(JSON.stringify({ subscribe: callSid }))
      setConnected(true)
    }

    ws.onmessage = (e) => {
      try {
        const data = JSON.parse(e.data) as LiveEvent
        onMessageRef.current(data)
      } catch (_) {
        // ignore
      }
    }

    ws.onclose = () => setConnected(false)
    ws.onerror = () => setConnected(false)

    return () => {
      ws.close()
      wsRef.current = null
      setConnected(false)
    }
  }, [callSid])

  return connected
}
