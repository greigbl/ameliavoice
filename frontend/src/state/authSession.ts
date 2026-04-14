import { create } from 'zustand'

const STORAGE_KEY = 'ameliavoice_authenticated'

function readStored(): boolean {
  try {
    return typeof sessionStorage !== 'undefined' && sessionStorage.getItem(STORAGE_KEY) === 'true'
  } catch {
    return false
  }
}

function writeStored(value: boolean) {
  try {
    if (typeof sessionStorage === 'undefined') return
    if (value) sessionStorage.setItem(STORAGE_KEY, 'true')
    else sessionStorage.removeItem(STORAGE_KEY)
  } catch {
    // ignore quota / private mode
  }
}

type AuthSessionState = {
  /** Client-side auth flag; mirrored in sessionStorage when true. Server cookie remains source of truth. */
  isAuthenticated: boolean
  setAuthenticated: (value: boolean) => void
  clearAuthenticated: () => void
}

export const useAuthSession = create<AuthSessionState>((set) => ({
  isAuthenticated: typeof window !== 'undefined' ? readStored() : false,
  setAuthenticated: (value: boolean) => {
    writeStored(value)
    set({ isAuthenticated: value })
  },
  clearAuthenticated: () => {
    writeStored(false)
    set({ isAuthenticated: false })
  },
}))

/** For axios interceptors (non-React). */
export function clearAuthSession() {
  useAuthSession.getState().clearAuthenticated()
}
