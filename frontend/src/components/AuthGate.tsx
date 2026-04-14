import { useEffect, useLayoutEffect, useState, useCallback } from 'react'
import { Outlet, useNavigate } from 'react-router-dom'
import Box from '@mui/material/Box'
import CircularProgress from '@mui/material/CircularProgress'
import { authMe, authLogout } from '../api/client'
import { useAuthSession } from '../state/authSession'
import { useSettings, type ClientIdOption } from '../state/settings'

export type AppAuthOutletContext = {
  /** When true, chat uses server session client_id; sidebar should lock the Client ID control. */
  sessionClientLocked: boolean
  onLogout: () => Promise<void>
  /** Server HIDE_SIDEBAR: hide settings sidebar from end users. */
  hideSidebar: boolean
}

export function AuthGate() {
  const navigate = useNavigate()
  const setClientId = useSettings((s) => s.setClientId)
  const isAuthenticated = useAuthSession((s) => s.isAuthenticated)
  const setAuthenticated = useAuthSession((s) => s.setAuthenticated)
  const clearAuthenticated = useAuthSession((s) => s.clearAuthenticated)
  const [loading, setLoading] = useState(true)
  const [authDisabled, setAuthDisabled] = useState(false)
  const [hideSidebar, setHideSidebar] = useState(false)

  useEffect(() => {
    let cancelled = false
    ;(async () => {
      try {
        const me = await authMe()
        if (cancelled) return
        setHideSidebar(me.hide_sidebar === true)
        if (me.auth_disabled) {
          setAuthDisabled(true)
          setAuthenticated(true)
        } else if (me.authenticated && me.client_id) {
          setAuthenticated(true)
          const cid = me.client_id as ClientIdOption
          if (['isuzu', 'monterey', 'maejima'].includes(cid)) {
            setClientId(cid)
          }
        } else {
          clearAuthenticated()
        }
      } catch {
        if (!cancelled) clearAuthenticated()
      } finally {
        if (!cancelled) setLoading(false)
      }
    })()
    return () => {
      cancelled = true
    }
  }, [setClientId, setAuthenticated, clearAuthenticated])

  useLayoutEffect(() => {
    if (loading || authDisabled) return
    if (!isAuthenticated) {
      navigate('/login', { replace: true })
    }
  }, [loading, authDisabled, isAuthenticated, navigate])

  const onLogout = useCallback(async () => {
    try {
      await authLogout()
    } catch {
      // still clear local session UX
    }
    clearAuthenticated()
    navigate('/login', { replace: true })
  }, [navigate, clearAuthenticated])

  if (loading) {
    return (
      <Box sx={{ display: 'flex', alignItems: 'center', justifyContent: 'center', height: '100vh' }}>
        <CircularProgress />
      </Box>
    )
  }

  if (!authDisabled && !isAuthenticated) {
    return null
  }

  const outletContext: AppAuthOutletContext = {
    sessionClientLocked: !authDisabled && isAuthenticated,
    onLogout,
    hideSidebar,
  }

  return <Outlet context={outletContext} />
}
