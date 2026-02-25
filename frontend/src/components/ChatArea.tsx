import { useTranslation } from 'react-i18next'
import Box from '@mui/material/Box'
import Paper from '@mui/material/Paper'
import TextField from '@mui/material/TextField'
import IconButton from '@mui/material/IconButton'
import Button from '@mui/material/Button'
import Typography from '@mui/material/Typography'
import Alert from '@mui/material/Alert'
import MicIcon from '@mui/icons-material/Mic'
import MicOffIcon from '@mui/icons-material/MicOff'
import SendIcon from '@mui/icons-material/Send'
import { useRef, useState, useEffect } from 'react'
import { useChat } from '../state/chat'
import { useVoiceConversation } from '../hooks/useVoiceConversation'
import { chat, type ChatMessage } from '../api/client'
import { useSettings } from '../state/settings'

export function ChatArea() {
  const { t } = useTranslation()
  const { messages, addMessage } = useChat()
  const { integration, language, verbosity } = useSettings()
  const {
    isListening,
    isProcessing,
    error,
    toggleVoice,
    clearError,
  } = useVoiceConversation()
  const [input, setInput] = useState('')
  const inputRef = useRef<HTMLInputElement>(null)
  const messagesEndRef = useRef<HTMLDivElement>(null)

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [messages, isListening, isProcessing])

  const handleSend = async () => {
    const text = input.trim()
    if (!text || isProcessing) return
    setInput('')
    addMessage({ role: 'user', content: text })
    const newMessages: ChatMessage[] = [
      ...messages,
      { role: 'user', content: text },
    ]
    try {
      const res = await chat(newMessages, integration, language, verbosity)
      addMessage({ role: 'assistant', content: res.message.content })
    } catch (e) {
      addMessage({
        role: 'assistant',
        content: t('error') + ': ' + (e instanceof Error ? e.message : String(e)),
      })
    }
  }

  return (
    <Box
      sx={{
        flex: 1,
        display: 'flex',
        flexDirection: 'column',
        minHeight: 0,
      }}
    >
      <Box
        sx={{
          flex: 1,
          overflow: 'auto',
          p: 2,
          display: 'flex',
          flexDirection: 'column',
          gap: 2,
        }}
      >
        {messages.length === 0 && !isListening && !isProcessing && (
          <Typography color="text.secondary" sx={{ alignSelf: 'center', mt: 4 }}>
            {t('placeholderInput')}
          </Typography>
        )}
        {messages.map((m, i) => (
          <Box
            key={i}
            sx={{
              alignSelf: m.role === 'user' ? 'flex-end' : 'flex-start',
              maxWidth: '85%',
            }}
          >
            <Typography
              variant="caption"
              color="text.secondary"
              sx={{ display: 'block', mb: 0.5 }}
            >
              {m.role === 'user' ? t('user') : t('agent')}
            </Typography>
            <Paper
              elevation={0}
              sx={{
                p: 1.5,
                bgcolor: m.role === 'user' ? 'primary.light' : 'grey.100',
                color: m.role === 'user' ? 'primary.contrastText' : 'text.primary',
                borderRadius: 2,
              }}
            >
              <Typography variant="body1" sx={{ whiteSpace: 'pre-wrap' }}>
                {m.content}
              </Typography>
              {m.role === 'assistant' && m.latency && (
                <Typography variant="caption" sx={{ mt: 1.5, display: 'block' }}>
                  <Typography component="span" sx={{ color: 'text.secondary', fontSize: '0.75rem' }}>
                    STT {m.latency.stt_ms} ms | LLM {m.latency.llm_ms} ms | TTS {m.latency.tts_ms} ms
                  </Typography>
                  <Typography component="span" sx={{ color: 'primary.main', fontWeight: 600, fontSize: '0.75rem' }}>
                    {' | '}Total {m.latency.stt_ms + m.latency.llm_ms + m.latency.tts_ms} ms
                  </Typography>
                </Typography>
              )}
            </Paper>
          </Box>
        ))}
        {isListening && (
          <Box sx={{ alignSelf: 'center', textAlign: 'center' }}>
            <Typography variant="body2" color="primary">
              {t('listening')}
            </Typography>
            <Typography variant="caption" color="text.secondary" display="block">
              {t('listeningHint')}
            </Typography>
          </Box>
        )}
        {isProcessing && (
          <Typography variant="body2" color="text.secondary" sx={{ alignSelf: 'center' }}>
            {t('processing')}
          </Typography>
        )}
        <div ref={messagesEndRef} />
      </Box>

      {error && (
        <Alert severity="error" onClose={clearError} sx={{ mx: 2, mt: 1 }}>
          {error}
        </Alert>
      )}

      <Paper
        elevation={2}
        sx={{
          p: 1.5,
          display: 'flex',
          gap: 1,
          alignItems: 'flex-end',
          borderRadius: 0,
        }}
      >
        <TextField
          inputRef={inputRef}
          fullWidth
          multiline
          maxRows={4}
          placeholder={t('placeholderInput')}
          value={input}
          onChange={(e) => setInput(e.target.value)}
          onKeyDown={(e) => {
            if (e.key === 'Enter' && !e.shiftKey) {
              e.preventDefault()
              handleSend()
            }
          }}
          size="small"
          disabled={isProcessing}
          sx={{ flex: 1 }}
        />
        <Button
          variant={isListening ? 'contained' : 'outlined'}
          color={isListening ? 'error' : 'primary'}
          onClick={toggleVoice}
          disabled={isProcessing}
          startIcon={isListening ? <MicOffIcon /> : <MicIcon />}
          sx={{ minWidth: 200 }}
        >
          {isListening ? t('stopVoiceConversation') : t('startVoiceConversation')}
        </Button>
        <IconButton
          color="primary"
          onClick={handleSend}
          disabled={!input.trim() || isProcessing}
          title={t('send')}
        >
          <SendIcon />
        </IconButton>
      </Paper>
    </Box>
  )
}
