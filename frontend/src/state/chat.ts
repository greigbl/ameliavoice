import { create } from 'zustand'
import type { ChatMessage } from '../api/client'

interface ChatState {
  messages: ChatMessage[]
  addMessage: (m: ChatMessage) => void
  setMessages: (m: ChatMessage[]) => void
  clearMessages: () => void
}

export const useChat = create<ChatState>((set) => ({
  messages: [],
  addMessage: (m) => set((s) => ({ messages: [...s.messages, m] })),
  setMessages: (m) => set({ messages: m }),
  clearMessages: () => set({ messages: [] }),
}))
