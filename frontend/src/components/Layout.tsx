import { useState } from 'react'
import Box from '@mui/material/Box'
import Button from '@mui/material/Button'
import IconButton from '@mui/material/IconButton'
import MenuIcon from '@mui/icons-material/Menu'
import ChevronLeftIcon from '@mui/icons-material/ChevronLeft'
import { Outlet, useOutletContext } from 'react-router-dom'
import type { AppAuthOutletContext } from './AuthGate'
import { Sidebar } from './Sidebar'

export function Layout() {
  const [sidebarOpen, setSidebarOpen] = useState(true)
  const auth = useOutletContext<AppAuthOutletContext | undefined>()
  const hideSidebar = auth?.hideSidebar ?? false
  const showHeaderLogout =
    hideSidebar && auth?.sessionClientLocked && auth?.onLogout
  const showTopBar = !hideSidebar || showHeaderLogout

  return (
    <Box sx={{ display: 'flex', height: '100vh', overflow: 'hidden' }}>
      {!hideSidebar && sidebarOpen && (
        <Box sx={{ position: 'relative' }}>
          <Sidebar
            sessionClientLocked={auth?.sessionClientLocked ?? false}
            onLogout={auth?.onLogout}
          />
          {/* Chevron overlay at top-right of sidebar to collapse it */}
          <IconButton
            size="small"
            onClick={() => setSidebarOpen(false)}
            sx={{
              position: 'absolute',
              top: 8,
              right: -14,
              bgcolor: 'background.paper',
              boxShadow: 1,
              '&:hover': { bgcolor: 'background.default' },
            }}
            aria-label="Hide settings panel"
          >
            <ChevronLeftIcon fontSize="small" />
          </IconButton>
        </Box>
      )}
      <Box
        sx={{
          flex: 1,
          display: 'flex',
          flexDirection: 'column',
          minWidth: 0,
          overflow: 'hidden',
          bgcolor: 'background.default',
        }}
      >
        {showTopBar && (
          <Box
            sx={{
              display: 'flex',
              alignItems: 'center',
              justifyContent: showHeaderLogout ? 'flex-end' : 'flex-start',
              px: 1,
              py: 0.5,
              borderBottom: 1,
              borderColor: 'divider',
              bgcolor: 'background.paper',
            }}
          >
            {!hideSidebar && !sidebarOpen && (
              <IconButton
                size="small"
                onClick={() => setSidebarOpen(true)}
                aria-label="Show settings panel"
              >
                <MenuIcon fontSize="small" />
              </IconButton>
            )}
            {showHeaderLogout && (
              <Button size="small" color="inherit" onClick={() => void auth.onLogout()}>
                Log out
              </Button>
            )}
          </Box>
        )}

        <Outlet />
      </Box>
    </Box>
  )
}
