import { useCallback, useEffect, useRef, useState } from 'react'
import { useSettings } from '../state/settings'
import { useChat } from '../state/chat'
import { transcribe, textToSpeech, chat, endVoiceSession, type ChatMessage } from '../api/client'
import { stripMarkdownForTTS } from '../utils/stripMarkdown'

// Voice activity: only auto-send after user has spoken, then paused (silence).
const SILENCE_RMS_THRESHOLD = 0.015   // below this = silence (time-domain RMS 0–1)
const SPEECH_RMS_THRESHOLD = 0.025   // above this = speaking (so we set hasSpoken)
const BARGE_IN_ENABLED = false       // when true, mic above threshold stops TTS; set false to avoid TTS stopping from noise
const BARGE_IN_RMS_THRESHOLD = 0.02  // above this while TTS playing = user interrupting, stop TTS (only if BARGE_IN_ENABLED)
const SILENCE_MS = 1000              // pause duration to trigger send (1 second)
const MIN_RECORDING_MS = 600         // minimum recording length before we accept silence

export function useVoiceConversation() {
  const [isListening, setIsListening] = useState(false)
  const [isProcessing, setIsProcessing] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const { language, integration, voiceModel, verbosity, continuousVoiceMode } = useSettings()
  const { messages, addMessage } = useChat()

  useEffect(() => {
    messagesRef.current = messages
  }, [messages])
  const streamRef = useRef<MediaStream | null>(null)
  const recorderRef = useRef<MediaRecorder | null>(null)
  const chunksRef = useRef<Blob[]>([])
  const analyserRef = useRef<AnalyserNode | null>(null)
  const silenceStartRef = useRef<number | null>(null)
  const hasSpokenRef = useRef(false)
  const animationRef = useRef<number | null>(null)
  const recordingStartRef = useRef<number | null>(null)
  const stopRecordingAndSendRef = useRef<{ fn: () => Promise<void> } | null>(null)
  const userRequestedStopRef = useRef(false)
  const messagesRef = useRef<ChatMessage[]>([])
  const currentTTSRef = useRef<{ audio: HTMLAudioElement; url: string } | null>(null)

  const languageCode = language === 'JA' ? 'ja-JP' : 'en-US'
  const asrModel = voiceModel

  const checkVolume = useCallback(() => {
    const analyser = analyserRef.current
    if (!analyser) return
    const data = new Float32Array(analyser.fftSize)
    analyser.getFloatTimeDomainData(data)
    let sumSq = 0
    for (let i = 0; i < data.length; i++) sumSq += data[i] * data[i]
    const rms = Math.sqrt(sumSq / data.length)
    // Barge-in: when enabled, user speaking over TTS stops playback (disabled by default to avoid TTS cutting off from noise)
    const tts = currentTTSRef.current
    if (BARGE_IN_ENABLED && tts && rms > BARGE_IN_RMS_THRESHOLD) {
      tts.audio.pause()
      tts.audio.currentTime = 0
      URL.revokeObjectURL(tts.url)
      currentTTSRef.current = null
    }
    if (rms > SPEECH_RMS_THRESHOLD) hasSpokenRef.current = true
    const recordingDuration = recordingStartRef.current
      ? Date.now() - recordingStartRef.current
      : 0
    const canSend =
      hasSpokenRef.current &&
      recordingDuration >= MIN_RECORDING_MS
    if (canSend && rms < SILENCE_RMS_THRESHOLD) {
      const start = silenceStartRef.current
      if (start == null) silenceStartRef.current = Date.now()
      else if (Date.now() - start >= SILENCE_MS) {
        stopRecordingAndSendRef.current?.fn()
        return
      }
    } else {
      silenceStartRef.current = null
    }
    animationRef.current = requestAnimationFrame(checkVolume)
  }, [])

  const restartRecording = useCallback(() => {
    const stream = streamRef.current
    if (!stream) return
    const mimeType = MediaRecorder.isTypeSupported('audio/webm;codecs=opus')
      ? 'audio/webm;codecs=opus'
      : 'audio/webm'
    const recorder = new MediaRecorder(stream, { mimeType })
    chunksRef.current = []
    recordingStartRef.current = Date.now()
    silenceStartRef.current = null
    hasSpokenRef.current = false
    recorder.ondataavailable = (e) => {
      if (e.data.size > 0) chunksRef.current.push(e.data)
    }
    recorder.start(200)
    recorderRef.current = recorder
    animationRef.current = requestAnimationFrame(checkVolume)
  }, [checkVolume])

  const stopRecordingAndSend = useCallback(async () => {
    const recorder = recorderRef.current
    const stream = streamRef.current
    const continuous = continuousVoiceMode
    const userStopped = userRequestedStopRef.current

    if (userStopped) {
      const tts = currentTTSRef.current
      if (tts) {
        tts.audio.pause()
        URL.revokeObjectURL(tts.url)
        currentTTSRef.current = null
      }
      if (recorder?.state === 'recording') recorder.stop()
      if (stream) {
        stream.getTracks().forEach((t) => t.stop())
        streamRef.current = null
      }
      setIsListening(false)
      if (animationRef.current != null) {
        cancelAnimationFrame(animationRef.current)
        animationRef.current = null
      }
      silenceStartRef.current = null
      hasSpokenRef.current = false
      chunksRef.current = []
      return
    }

    if (!recorder || recorder.state !== 'recording') return
    recorder.stop()
    if (!continuous && stream) {
      stream.getTracks().forEach((t) => t.stop())
      streamRef.current = null
    }
    if (!continuous) setIsListening(false)
    if (animationRef.current != null) {
      cancelAnimationFrame(animationRef.current)
      animationRef.current = null
    }
    silenceStartRef.current = null
    hasSpokenRef.current = false
    const chunks = chunksRef.current
    chunksRef.current = []
    const doRestart = () => {
      if (continuous && streamRef.current && !userRequestedStopRef.current) restartRecording()
    }
    if (chunks.length === 0) {
      doRestart()
      return
    }
    const blob = new Blob(chunks, { type: 'audio/webm' })
    const duration = recordingStartRef.current
      ? Date.now() - recordingStartRef.current
      : 0
    if (duration < MIN_RECORDING_MS) {
      doRestart()
      return
    }
    setIsProcessing(true)
    setError(null)
    let endConversation = false
    try {
      const tStt0 = performance.now()
      const result = await transcribe(blob, language, asrModel)
      const stt_ms = Math.round(performance.now() - tStt0)
      const userText = result.text?.trim()
      if (!userText) {
        setIsProcessing(false)
        if (continuous) doRestart()
        return
      }
      addMessage({ role: 'user', content: userText })
      const currentMessages = messagesRef.current
      const newMessages: ChatMessage[] = [
        ...currentMessages,
        { role: 'user', content: userText },
      ]
      const tLlm0 = performance.now()
      //Put a call to Fred's backend here.
      
      const chatRes = await chat(newMessages, integration, language, verbosity)
      //BUt still need to handle object differences; request in / response out.
      
      const llm_ms = Math.round(performance.now() - tLlm0)
      const assistantContent = chatRes.message.content
      endConversation = Boolean(chatRes.end_conversation)
      const plainForTTS = stripMarkdownForTTS(assistantContent)
      const tTts0 = performance.now()
      const audioBlob = await textToSpeech(
        plainForTTS || assistantContent,
        languageCode
      )
      const tts_ms = Math.round(performance.now() - tTts0)
      addMessage({
        role: 'assistant',
        content: assistantContent,
        latency: { stt_ms, llm_ms, tts_ms },
      })
      const url = URL.createObjectURL(audioBlob)
      const audio = new Audio(url)
      currentTTSRef.current = { audio, url }
      audio.onended = () => {
        URL.revokeObjectURL(url)
        if (currentTTSRef.current?.url === url) currentTTSRef.current = null
        if (endConversation) {
          endVoiceSession().then(() => {
            userRequestedStopRef.current = true
            stopRecordingAndSendRef.current?.fn()
          })
        } else if (continuous) {
          const rec = recorderRef.current
          if (!rec || rec.state !== 'recording') doRestart()
        }
      }
      if (endConversation) {
        await audio.play()
      } else if (continuous) {
        audio.play()
        // Do not restart recording here — wait for TTS to finish (audio.onended).
        // Otherwise the mic picks up speaker output and STT transcribes the assistant's voice.
      } else {
        await audio.play()
        doRestart()
      }
    } catch (e) {
      const msg = e instanceof Error ? e.message : String(e)
      setError(msg)
    } finally {
      setIsProcessing(false)
      if (continuous && !currentTTSRef.current && !endConversation) doRestart()
    }
  }, [
    integration,
    language,
    verbosity,
    languageCode,
    asrModel,
    addMessage,
    continuousVoiceMode,
    restartRecording,
  ])

  useEffect(() => {
    stopRecordingAndSendRef.current = { fn: stopRecordingAndSend }
    return () => {
      stopRecordingAndSendRef.current = null
    }
  }, [stopRecordingAndSend])

  const startListening = useCallback(async () => {
    userRequestedStopRef.current = false
    setError(null)
    if (!navigator.mediaDevices?.getUserMedia) {
      setError(
        'Microphone access is not available. Use HTTPS or open the app from localhost (browsers block microphone on plain HTTP from other hosts).'
      )
      return
    }
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true })
      streamRef.current = stream
      const ac = new AudioContext()
      const source = ac.createMediaStreamSource(stream)
      const analyser = ac.createAnalyser()
      analyser.fftSize = 256
      source.connect(analyser)
      analyserRef.current = analyser
      const mimeType = MediaRecorder.isTypeSupported('audio/webm;codecs=opus')
        ? 'audio/webm;codecs=opus'
        : 'audio/webm'
      const recorder = new MediaRecorder(stream, { mimeType })
      chunksRef.current = []
      recorder.ondataavailable = (e) => {
        if (e.data.size > 0) chunksRef.current.push(e.data)
      }
      recorder.start(200)
      recorderRef.current = recorder
      recordingStartRef.current = Date.now()
      silenceStartRef.current = null
      hasSpokenRef.current = false
      setIsListening(true)
      requestAnimationFrame(checkVolume)
    } catch (e) {
      const msg = e instanceof Error ? e.message : String(e)
      setError(msg)
    }
  }, [checkVolume])

  const stopListening = useCallback(() => {
    userRequestedStopRef.current = true
    stopRecordingAndSend()
  }, [stopRecordingAndSend])

  const toggleVoice = useCallback(() => {
    if (isListening) stopListening()
    else startListening()
  }, [isListening, startListening, stopListening])

  return {
    isListening,
    isProcessing,
    error,
    toggleVoice,
    clearError: () => setError(null),
  }
}
