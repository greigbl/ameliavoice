import Box from '@mui/material/Box'
import Typography from '@mui/material/Typography'
import { useTranslation } from 'react-i18next'
import { ChatArea } from '../components/ChatArea'

export function ChatPage() {
  const { t } = useTranslation()

  return (
    <Box sx={{ display: 'flex', flexDirection: 'column', flex: 1, minHeight: 0 }}>
      <Box
        sx={{
          py: 1,
          px: 2,
          borderBottom: 1,
          borderColor: 'divider',
          bgcolor: 'background.paper',
        }}
      >
        <Typography variant="h6" color="text.primary">
          {t('appTitle')}
        </Typography>
      </Box>
      <ChatArea />
    </Box>
  )
}
