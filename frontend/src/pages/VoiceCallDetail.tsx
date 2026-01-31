import { useCallback, useEffect, useState } from 'react'
import Box from '@mui/material/Box'
import Chip from '@mui/material/Chip'
import Paper from '@mui/material/Paper'
import Typography from '@mui/material/Typography'
import { useParams, useNavigate } from 'react-router-dom'
import { getVoiceCall, type VoiceCallDetail, type VoiceCallTurn } from '../api/client'
import { useVoiceCallLive, type LiveEvent } from '../hooks/useVoiceCallLive'

function formatTime(ts: number | null): string {
  if (ts == null) return '—'
  const d = new Date(ts * 1000)
  return d.toLocaleString()
}

/** Partial turn while pipeline is running (STT/LLM/TTS updates stream in). */
interface CurrentTurnPartial {
  user_text?: string
  assistant_text?: string
  stt_ms?: number
  llm_ms?: number
  tts_ms?: number
}

function TurnRow({ turn }: { turn: VoiceCallTurn }) {
  return (
    <Box sx={{ mb: 2 }}>
      <Paper variant="outlined" sx={{ p: 2, bgcolor: 'grey.50' }}>
        <Typography variant="caption" color="text.secondary" sx={{ display: 'block', mb: 0.5 }}>
          User
        </Typography>
        <Typography variant="body1" sx={{ whiteSpace: 'pre-wrap' }}>
          {turn.user_text || '—'}
        </Typography>
      </Paper>
      <Paper variant="outlined" sx={{ p: 2, mt: 1, bgcolor: 'grey.100' }}>
        <Typography variant="caption" color="text.secondary" sx={{ display: 'block', mb: 0.5 }}>
          Agent
        </Typography>
        <Typography variant="body1" sx={{ whiteSpace: 'pre-wrap' }}>
          {turn.assistant_text || '—'}
        </Typography>
        <Box sx={{ mt: 1.5, display: 'flex', flexWrap: 'wrap', gap: 1 }}>
          <Chip size="small" label={`STT ${turn.stt_ms} ms`} variant="outlined" />
          <Chip size="small" label={`LLM ${turn.llm_ms} ms`} variant="outlined" />
          <Chip size="small" label={`TTS ${turn.tts_ms} ms`} variant="outlined" />
          <Chip
            size="small"
            label={`Total ${(turn.stt_ms + turn.llm_ms + turn.tts_ms).toFixed(0)} ms`}
            color="primary"
            variant="filled"
          />
        </Box>
      </Paper>
    </Box>
  )
}

/** In-progress turn: show whatever we have so far (user text, then agent text, then latencies). */
function CurrentTurnRow({ turn }: { turn: CurrentTurnPartial }) {
  const hasUser = turn.user_text != null && turn.user_text !== ''
  const hasAgent = turn.assistant_text != null && turn.assistant_text !== ''
  const totalMs =
    (turn.stt_ms ?? 0) + (turn.llm_ms ?? 0) + (turn.tts_ms ?? 0)
  const hasLatency = totalMs > 0

  return (
    <Box sx={{ mb: 2 }}>
      <Paper variant="outlined" sx={{ p: 2, bgcolor: 'grey.50', borderStyle: 'dashed' }}>
        <Typography variant="caption" color="text.secondary" sx={{ display: 'block', mb: 0.5 }}>
          User
        </Typography>
        <Typography variant="body1" sx={{ whiteSpace: 'pre-wrap' }}>
          {hasUser ? turn.user_text : '…'}
        </Typography>
      </Paper>
      <Paper variant="outlined" sx={{ p: 2, mt: 1, bgcolor: 'grey.100', borderStyle: 'dashed' }}>
        <Typography variant="caption" color="text.secondary" sx={{ display: 'block', mb: 0.5 }}>
          Agent
        </Typography>
        <Typography variant="body1" sx={{ whiteSpace: 'pre-wrap' }}>
          {hasAgent ? turn.assistant_text : '…'}
        </Typography>
        {(turn.stt_ms != null || turn.llm_ms != null || turn.tts_ms != null) && (
          <Box sx={{ mt: 1.5, display: 'flex', flexWrap: 'wrap', gap: 1 }}>
            {turn.stt_ms != null && (
              <Chip size="small" label={`STT ${turn.stt_ms} ms`} variant="outlined" />
            )}
            {turn.llm_ms != null && (
              <Chip size="small" label={`LLM ${turn.llm_ms} ms`} variant="outlined" />
            )}
            {turn.tts_ms != null && (
              <Chip size="small" label={`TTS ${turn.tts_ms} ms`} variant="outlined" />
            )}
            {hasLatency && (
              <Chip
                size="small"
                label={`Total ${totalMs.toFixed(0)} ms`}
                color="primary"
                variant="filled"
              />
            )}
          </Box>
        )}
      </Paper>
    </Box>
  )
}

export function VoiceCallDetail() {
  const { callSid } = useParams<{ callSid: string }>()
  const navigate = useNavigate()
  const [call, setCall] = useState<VoiceCallDetail | null>(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)
  const [turns, setTurns] = useState<VoiceCallTurn[]>([])
  const [currentTurn, setCurrentTurn] = useState<CurrentTurnPartial | null>(null)

  const handleLiveMessage = useCallback((ev: LiveEvent) => {
    const { event: evName, payload } = ev
    switch (evName) {
      case 'stt_done': {
        const user_text = (payload.user_text as string) ?? ''
        const stt_ms = (payload.stt_ms as number) ?? 0
        setCurrentTurn((prev) => ({ ...prev, user_text, stt_ms }))
        break
      }
      case 'llm_done': {
        const assistant_text = (payload.assistant_text as string) ?? ''
        const llm_ms = (payload.llm_ms as number) ?? 0
        setCurrentTurn((prev) => ({ ...prev, assistant_text, llm_ms }))
        break
      }
      case 'tts_start': {
        setCurrentTurn((prev) => (prev ? prev : {}))
        break
      }
      case 'tts_done': {
        const tts_ms = (payload.tts_ms as number) ?? 0
        setCurrentTurn((prev) => {
          if (!prev) return null
          const full: VoiceCallTurn = {
            user_text: prev.user_text ?? '',
            assistant_text: prev.assistant_text ?? '',
            stt_ms: prev.stt_ms ?? 0,
            llm_ms: prev.llm_ms ?? 0,
            tts_ms,
          }
          setTurns((t) => [...t, full])
          return null
        })
        break
      }
      default:
        break
    }
  }, [])

  useVoiceCallLive(callSid, handleLiveMessage)

  useEffect(() => {
    if (!callSid) return
    let cancelled = false
    setLoading(true)
    setError(null)
    getVoiceCall(callSid)
      .then((data) => {
        if (!cancelled) {
          setCall(data)
          setTurns(data.turns)
        }
      })
      .catch((e) => {
        if (!cancelled)
          setError(e?.response?.data?.detail ?? e?.message ?? 'Failed to load call')
      })
      .finally(() => {
        if (!cancelled) setLoading(false)
      })
    return () => {
      cancelled = true
    }
  }, [callSid])

  if (loading && !call) {
    return (
      <Box sx={{ p: 2 }}>
        <Typography>Loading…</Typography>
      </Box>
    )
  }
  if (error || !call) {
    return (
      <Box sx={{ p: 2 }}>
        <Typography color="error">{error ?? 'Call not found'}</Typography>
        <Typography
          component="button"
          variant="body2"
          onClick={() => navigate('/calls')}
          sx={{ mt: 1, textDecoration: 'underline', cursor: 'pointer' }}
        >
          Back to calls
        </Typography>
      </Box>
    )
  }

  return (
    <Box
      sx={{
        p: 2,
        flex: 1,
        overflow: 'auto',
        display: 'flex',
        flexDirection: 'column',
        minHeight: 0,
      }}
    >
      <Typography
        component="button"
        variant="body2"
        onClick={() => navigate('/calls')}
        sx={{
          mb: 2,
          display: 'block',
          textDecoration: 'underline',
          cursor: 'pointer',
          color: 'primary.main',
        }}
      >
        ← Back to calls
      </Typography>
      <Typography variant="h6" sx={{ mb: 1 }}>
        Call transcript
      </Typography>
      <Typography variant="body2" color="text.secondary" sx={{ mb: 2, fontFamily: 'monospace' }}>
        {call.call_sid}
      </Typography>
      <Typography variant="body2" color="text.secondary" sx={{ mb: 2 }}>
        Started {formatTime(call.start_time)} · Ended {formatTime(call.end_time)} · {turns.length}
        turn(s){currentTurn != null ? ' (live)' : ''}
      </Typography>
      {turns.length === 0 && currentTurn == null ? (
        <Typography color="text.secondary">No turns in this call yet.</Typography>
      ) : (
        <>
          {turns.map((turn, i) => (
            <TurnRow key={i} turn={turn} />
          ))}
          {currentTurn != null && <CurrentTurnRow turn={currentTurn} />}
        </>
      )}
    </Box>
  )
}
