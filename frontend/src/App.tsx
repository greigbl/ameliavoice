import { Routes, Route } from 'react-router-dom'
import { AuthGate } from './components/AuthGate'
import { Layout } from './components/Layout'
import { ChatPage } from './pages/ChatPage'
import { LoginPage } from './pages/LoginPage'
import { VoiceCallsList } from './pages/VoiceCallsList'
import { VoiceCallDetail } from './pages/VoiceCallDetail'

export default function App() {
  return (
    <Routes>
      <Route path="/login" element={<LoginPage />} />
      <Route element={<AuthGate />}>
        <Route path="/" element={<Layout />}>
          <Route index element={<ChatPage />} />
          <Route path="calls" element={<VoiceCallsList />} />
          <Route path="calls/:callSid" element={<VoiceCallDetail />} />
        </Route>
      </Route>
    </Routes>
  )
}
