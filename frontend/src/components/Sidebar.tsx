import { useTranslation } from 'react-i18next'
import Box from '@mui/material/Box'
import Checkbox from '@mui/material/Checkbox'
import FormControlLabel from '@mui/material/FormControlLabel'
import FormControl from '@mui/material/FormControl'
import InputLabel from '@mui/material/InputLabel'
import MenuItem from '@mui/material/MenuItem'
import Select from '@mui/material/Select'
import Typography from '@mui/material/Typography'
import Link from '@mui/material/Link'
import { Link as RouterLink, useLocation } from 'react-router-dom'
import { useSettings, type Integration, type LanguageOption, type VoiceModelOption, type VerbosityOption } from '../state/settings'

const SIDEBAR_WIDTH = 260

export function Sidebar() {
  const { t } = useTranslation()
  const location = useLocation()
  const {
    domain,
    integration,
    language,
    voiceModel,
    verbosity,
    continuousVoiceMode,
    setDomain,
    setIntegration,
    setLanguage,
    setVoiceModel,
    setVerbosity,
    setContinuousVoiceMode,
  } = useSettings()

  return (
    <Box
      sx={{
        width: SIDEBAR_WIDTH,
        minWidth: SIDEBAR_WIDTH,
        height: '100vh',
        borderRight: 1,
        borderColor: 'divider',
        bgcolor: 'background.paper',
        p: 2,
        display: 'flex',
        flexDirection: 'column',
        gap: 2,
      }}
    >
      <Box sx={{ display: 'flex', flexDirection: 'column', gap: 0.5, mb: 2 }}>
        <Link component={RouterLink} to="/" underline={location.pathname === '/' ? 'always' : 'hover'} color="inherit">
          Chat
        </Link>
        <Link component={RouterLink} to="/calls" underline={location.pathname.startsWith('/calls') ? 'always' : 'hover'} color="inherit">
          Voice calls
        </Link>
      </Box>
      <Typography variant="subtitle2" color="text.secondary" sx={{ fontWeight: 600 }}>
        {t('domain')}
      </Typography>
      <FormControl size="small" fullWidth>
        <InputLabel id="domain-label">{t('domain')}</InputLabel>
        <Select
          labelId="domain-label"
          label={t('domain')}
          value={domain}
          onChange={(e) => setDomain(e.target.value)}
        >
          <MenuItem value="">—</MenuItem>
          <MenuItem value="default">Default</MenuItem>
        </Select>
      </FormControl>

      <Typography variant="subtitle2" color="text.secondary" sx={{ fontWeight: 600 }}>
        {t('integration')}
      </Typography>
      <FormControl size="small" fullWidth>
        <InputLabel id="integration-label">{t('integration')}</InputLabel>
        <Select
          labelId="integration-label"
          label={t('integration')}
          value={integration}
          onChange={(e) => setIntegration(e.target.value as Integration)}
        >
          <MenuItem value="openai">{t('integrationOpenAI')}</MenuItem>
          <MenuItem value="amelia">{t('integrationAmelia')}</MenuItem>
        </Select>
      </FormControl>

      <Typography variant="subtitle2" color="text.secondary" sx={{ fontWeight: 600 }}>
        {t('language')}
      </Typography>
      <FormControl size="small" fullWidth>
        <InputLabel id="language-label">{t('language')}</InputLabel>
        <Select
          labelId="language-label"
          label={t('language')}
          value={language}
          onChange={(e) => setLanguage(e.target.value as LanguageOption)}
        >
          <MenuItem value="EN">{t('languageEN')}</MenuItem>
          <MenuItem value="JA">{t('languageJA')}</MenuItem>
        </Select>
      </FormControl>

      <Typography variant="subtitle2" color="text.secondary" sx={{ fontWeight: 600 }}>
        {t('voiceModel')}
      </Typography>
      <FormControl size="small" fullWidth>
        <InputLabel id="voice-model-label">{t('voiceModel')}</InputLabel>
        <Select
          labelId="voice-model-label"
          label={t('voiceModel')}
          value={voiceModel}
          onChange={(e) => setVoiceModel(e.target.value as VoiceModelOption)}
        >
          <MenuItem value="google">{t('voiceModelGoogle')}</MenuItem>
          <MenuItem value="whisper">{t('voiceModelWhisper')}</MenuItem>
        </Select>
      </FormControl>

      <Typography variant="subtitle2" color="text.secondary" sx={{ fontWeight: 600 }}>
        {t('verbosity')}
      </Typography>
      <FormControl size="small" fullWidth>
        <InputLabel id="verbosity-label">{t('verbosity')}</InputLabel>
        <Select
          labelId="verbosity-label"
          label={t('verbosity')}
          value={verbosity}
          onChange={(e) => setVerbosity(e.target.value as VerbosityOption)}
        >
          <MenuItem value="brief">{t('verbosityBrief')}</MenuItem>
          <MenuItem value="normal">{t('verbosityNormal')}</MenuItem>
          <MenuItem value="detailed">{t('verbosityDetailed')}</MenuItem>
        </Select>
      </FormControl>

      <Typography variant="subtitle2" color="text.secondary" sx={{ fontWeight: 600, mt: 1 }}>
        {t('voiceMode')}
      </Typography>
      <FormControlLabel
        control={
          <Checkbox
            checked={continuousVoiceMode}
            onChange={(_, checked) => setContinuousVoiceMode(checked)}
            size="small"
          />
        }
        label={t('continuousVoiceMode')}
        sx={{ mt: 0.5 }}
      />
    </Box>
  )
}
