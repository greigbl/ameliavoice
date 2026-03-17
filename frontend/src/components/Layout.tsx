import { useState } from 'react'
import Box from '@mui/material/Box'
import IconButton from '@mui/material/IconButton'
import MenuIcon from '@mui/icons-material/Menu'
import ChevronLeftIcon from '@mui/icons-material/ChevronLeft'
import { Outlet } from 'react-router-dom'
import { Sidebar } from './Sidebar'

export function Layout() {
  const [sidebarOpen, setSidebarOpen] = useState(true)

  return (
    <Box sx={{ display: 'flex', height: '100vh', overflow: 'hidden' }}>
      {sidebarOpen && (
        <Box sx={{ position: 'relative' }}>
          <Sidebar />
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
        {/* Top bar with toggle when sidebar is collapsed */}
        <Box
          sx={{
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'flex-start',
            px: 1,
            py: 0.5,
            borderBottom: 1,
            borderColor: 'divider',
            bgcolor: 'background.paper',
          }}
        >
          {!sidebarOpen && (
            <IconButton
              size="small"
              onClick={() => setSidebarOpen(true)}
              aria-label="Show settings panel"
            >
              <MenuIcon fontSize="small" />
            </IconButton>
          )}
        </Box>

        <Outlet />
      </Box>
    </Box>
  )
}
