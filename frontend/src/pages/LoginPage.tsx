import { useEffect, useState } from 'react'
import { useNavigate } from 'react-router-dom'
import Alert from '@mui/material/Alert'
import Box from '@mui/material/Box'
import Button from '@mui/material/Button'
import Paper from '@mui/material/Paper'
import TextField from '@mui/material/TextField'
import Typography from '@mui/material/Typography'
import { authLogin, authMe } from '../api/client'
import { useAuthSession } from '../state/authSession'
import { useSettings, type ClientIdOption } from '../state/settings'

export function LoginPage() {
  const navigate = useNavigate()
  const setClientId = useSettings((s) => s.setClientId)
  const setAuthenticated = useAuthSession((s) => s.setAuthenticated)
  const [username, setUsername] = useState('')
  const [password, setPassword] = useState('')
  const [error, setError] = useState<string | null>(null)
  const [submitting, setSubmitting] = useState(false)
  /** Server has AUTH_JWT_SECRET unset — sign-in is not enforced. */
  const [authDisabledOnServer, setAuthDisabledOnServer] = useState(false)

  useEffect(() => {
    let cancelled = false
    ;(async () => {
      try {
        const me = await authMe()
        if (cancelled) return
        if (me.auth_disabled) {
          setAuthDisabledOnServer(true)
          return
        }
        if (me.authenticated) {
          setAuthenticated(true)
          navigate('/', { replace: true })
        }
      } catch {
        // ignore
      }
    })()
    return () => {
      cancelled = true
    }
  }, [navigate, setAuthenticated])

  function continueWithoutSignIn() {
    setAuthenticated(true)
    navigate('/', { replace: true })
  }

  async function handleSubmit(e: React.FormEvent) {
    e.preventDefault()
    setError(null)
    setSubmitting(true)
    try {
      const res = await authLogin(username.trim(), password)
      const cid = res.client_id as ClientIdOption
      if (['isuzu', 'monterey', 'maejima'].includes(cid)) {
        setClientId(cid)
      }
      setAuthenticated(true)
      navigate('/', { replace: true })
    } catch (err: unknown) {
      const msg =
        err && typeof err === 'object' && 'response' in err
          ? (err as { response?: { data?: { detail?: string } } }).response?.data?.detail
          : null
      setError(typeof msg === 'string' ? msg : 'Login failed')
    } finally {
      setSubmitting(false)
    }
  }

  return (
    <Box
      sx={{
        minHeight: '100vh',
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'center',
        bgcolor: 'background.default',
        p: 2,
      }}
    >
      <Paper elevation={2} sx={{ p: 3, maxWidth: 360, width: '100%' }}>
        <Typography variant="h6" component="h1" gutterBottom>
          Sign in
        </Typography>
        <Typography variant="body2" color="text.secondary" sx={{ mb: 2 }}>
          Use the username and password configured for your tenant.
        </Typography>
        {authDisabledOnServer && (
          <Alert severity="info" sx={{ mb: 2 }}>
            The server does not have authentication enabled (set <code>AUTH_JWT_SECRET</code> in the API
            environment). Sign-in is optional; you can open the app without credentials.
          </Alert>
        )}
        {authDisabledOnServer && (
          <Button variant="contained" fullWidth sx={{ mb: 2 }} onClick={() => continueWithoutSignIn()}>
            Continue without signing in
          </Button>
        )}
        <Box
          component="form"
          onSubmit={handleSubmit}
          sx={{ display: 'flex', flexDirection: 'column', gap: 2 }}
        >
          <TextField
            label="Username"
            name="username"
            autoComplete="username"
            value={username}
            onChange={(e) => setUsername(e.target.value)}
            fullWidth
            required
            disabled={submitting || authDisabledOnServer}
          />
          <TextField
            label="Password"
            name="password"
            type="password"
            autoComplete="current-password"
            value={password}
            onChange={(e) => setPassword(e.target.value)}
            fullWidth
            required
            disabled={submitting || authDisabledOnServer}
          />
          {error && (
            <Typography variant="body2" color="error">
              {error}
            </Typography>
          )}
          <Button type="submit" variant="outlined" disabled={submitting || authDisabledOnServer}>
            {submitting ? 'Signing in…' : 'Sign in'}
          </Button>
        </Box>
      </Paper>
    </Box>
  )
}
