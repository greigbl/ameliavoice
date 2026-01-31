import { create } from 'zustand'
import { persist } from 'zustand/middleware'
import i18n from '../i18n'

export type Integration = 'openai' | 'amelia'
export type LanguageOption = 'EN' | 'JA'
export type VoiceModelOption = 'google' | 'whisper'
export type VerbosityOption = 'brief' | 'normal' | 'detailed'

interface SettingsState {
  domain: string
  integration: Integration
  language: LanguageOption
  voiceModel: VoiceModelOption
  verbosity: VerbosityOption
  continuousVoiceMode: boolean
  setDomain: (v: string) => void
  setIntegration: (v: Integration) => void
  setLanguage: (v: LanguageOption) => void
  setVoiceModel: (v: VoiceModelOption) => void
  setVerbosity: (v: VerbosityOption) => void
  setContinuousVoiceMode: (v: boolean) => void
}

const STORAGE_KEY = 'ameliavoice-settings'

export const useSettings = create<SettingsState>()(
  persist(
    (set) => ({
      domain: '',
      integration: 'openai',
      language: 'EN',
      voiceModel: 'google',
      verbosity: 'normal',
      continuousVoiceMode: true,
      setDomain: (v) => set({ domain: v }),
      setIntegration: (v) => set({ integration: v }),
      setLanguage: (v) => {
        set({ language: v })
        i18n.changeLanguage(v === 'JA' ? 'ja' : 'en')
      },
      setVoiceModel: (v) => set({ voiceModel: v }),
      setVerbosity: (v) => set({ verbosity: v }),
      setContinuousVoiceMode: (v) => set({ continuousVoiceMode: v }),
    }),
    {
      name: STORAGE_KEY,
      partialize: (state) => ({
        domain: state.domain,
        integration: state.integration,
        language: state.language,
        voiceModel: state.voiceModel,
        verbosity: state.verbosity,
        continuousVoiceMode: state.continuousVoiceMode,
      }),
      onRehydrateStorage: () => (state, err) => {
        if (!err && state?.language) {
          i18n.changeLanguage(state.language === 'JA' ? 'ja' : 'en')
        }
      },
    }
  )
)
