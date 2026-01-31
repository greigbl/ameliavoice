import { useEffect, useState } from 'react'
import Box from '@mui/material/Box'
import Paper from '@mui/material/Paper'
import Table from '@mui/material/Table'
import TableBody from '@mui/material/TableBody'
import TableCell from '@mui/material/TableCell'
import TableContainer from '@mui/material/TableContainer'
import TableHead from '@mui/material/TableHead'
import TableRow from '@mui/material/TableRow'
import Typography from '@mui/material/Typography'
import { useNavigate } from 'react-router-dom'
import { getVoiceCalls, type VoiceCallSummary } from '../api/client'

const REFRESH_MS = 5000

function formatTime(ts: number | null): string {
  if (ts == null) return '—'
  const d = new Date(ts * 1000)
  return d.toLocaleString()
}

export function VoiceCallsList() {
  const [calls, setCalls] = useState<VoiceCallSummary[]>([])
  const [loading, setLoading] = useState(true)
  const navigate = useNavigate()

  const fetchCalls = async () => {
    try {
      const data = await getVoiceCalls()
      setCalls(data)
    } catch (e) {
      console.error('Failed to fetch voice calls', e)
    } finally {
      setLoading(false)
    }
  }

  useEffect(() => {
    fetchCalls()
    const id = setInterval(fetchCalls, REFRESH_MS)
    return () => clearInterval(id)
  }, [])

  return (
    <Box sx={{ p: 2, flex: 1, overflow: 'auto', display: 'flex', flexDirection: 'column', minHeight: 0 }}>
      <Typography variant="h6" sx={{ mb: 2 }}>
        Voice calls
      </Typography>
      <Typography variant="body2" color="text.secondary" sx={{ mb: 2 }}>
        Auto-refreshes every {REFRESH_MS / 1000}s. Click a row to open the transcript.
      </Typography>
      <TableContainer component={Paper}>
        <Table size="small" stickyHeader>
          <TableHead>
            <TableRow>
              <TableCell>Call SID</TableCell>
              <TableCell>Started</TableCell>
              <TableCell>Ended</TableCell>
              <TableCell align="right">Turns</TableCell>
            </TableRow>
          </TableHead>
          <TableBody>
            {loading && calls.length === 0 ? (
              <TableRow>
                <TableCell colSpan={4}>Loading…</TableCell>
              </TableRow>
            ) : calls.length === 0 ? (
              <TableRow>
                <TableCell colSpan={4}>No calls yet. Place a call to the Twilio number to see it here.</TableCell>
              </TableRow>
            ) : (
              calls.map((row) => (
                <TableRow
                  key={row.call_sid}
                  hover
                  sx={{ cursor: 'pointer' }}
                  onClick={() => navigate(`/calls/${row.call_sid}`)}
                >
                  <TableCell sx={{ fontFamily: 'monospace', fontSize: '0.85rem' }}>
                    {row.call_sid}
                  </TableCell>
                  <TableCell>{formatTime(row.start_time)}</TableCell>
                  <TableCell>{formatTime(row.end_time)}</TableCell>
                  <TableCell align="right">{row.turn_count}</TableCell>
                </TableRow>
              ))
            )}
          </TableBody>
        </Table>
      </TableContainer>
    </Box>
  )
}
