import { useCallback } from 'react'
import { create } from 'zustand'
import { persist } from 'zustand/middleware'
import type { ChatMessage } from '../api/client'
import type { ClientIdOption } from './settings'
import { useSettings } from './settings'

const CLIENT_KEYS: ClientIdOption[] = ['isuzu', 'monterey', 'maejima']

function emptyThreads(): Record<ClientIdOption, ChatMessage[]> {
  return { isuzu: [], monterey: [], maejima: [] }
}

type ChatStore = {
  byClientId: Record<ClientIdOption, ChatMessage[]>
  addMessage: (clientId: ClientIdOption, m: ChatMessage) => void
  setMessages: (clientId: ClientIdOption, m: ChatMessage[]) => void
  /** Pass clientId to clear one tenant; omit to clear all three. */
  clearMessages: (clientId?: ClientIdOption) => void
}

export const useChatStore = create<ChatStore>()(
  persist(
    (set) => ({
      byClientId: emptyThreads(),
      addMessage: (clientId, m) =>
        set((s) => ({
          byClientId: {
            ...s.byClientId,
            [clientId]: [...(s.byClientId[clientId] ?? []), m],
          },
        })),
      setMessages: (clientId, messages) =>
        set((s) => ({
          byClientId: { ...s.byClientId, [clientId]: messages },
        })),
      clearMessages: (clientId) =>
        set((s) =>
          clientId == null
            ? { byClientId: emptyThreads() }
            : { byClientId: { ...s.byClientId, [clientId]: [] } },
        ),
    }),
    {
      name: 'ameliavoice-chat',
      version: 2,
      partialize: (state) => ({ byClientId: state.byClientId }),
      migrate: (persisted, version) => {
        const v = version ?? 0
        if (v < 2 && persisted && typeof persisted === 'object') {
          const p = persisted as {
            messages?: ChatMessage[]
            byClientId?: Partial<Record<ClientIdOption, ChatMessage[]>>
          }
          if (p.byClientId && typeof p.byClientId === 'object') {
            const merged = emptyThreads()
            for (const k of CLIENT_KEYS) {
              merged[k] = Array.isArray(p.byClientId[k]) ? p.byClientId[k]! : []
            }
            return { byClientId: merged }
          }
          if (Array.isArray(p.messages)) {
            return {
              byClientId: {
                isuzu: p.messages,
                monterey: [],
                maejima: [],
              },
            }
          }
        }
        return persisted as { byClientId: Record<ClientIdOption, ChatMessage[]> }
      },
    },
  ),
)

/**
 * Messages and mutators scoped to the current settings `clientId` (synced from login when auth is on).
 * History is persisted per tenant (isuzu | monterey | maejima).
 */
export function useChat() {
  const clientId = useSettings((s) => s.clientId)
  const byClientId = useChatStore((s) => s.byClientId)
  const messages = byClientId[clientId] ?? []

  const addMessage = useCallback(
    (m: ChatMessage) => {
      useChatStore.getState().addMessage(clientId, m)
    },
    [clientId],
  )

  const setMessages = useCallback(
    (m: ChatMessage[]) => {
      useChatStore.getState().setMessages(clientId, m)
    },
    [clientId],
  )

  const clearMessages = useCallback(() => {
    useChatStore.getState().clearMessages(clientId)
  }, [clientId])

  return { messages, addMessage, setMessages, clearMessages }
}
