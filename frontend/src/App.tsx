import { Routes, Route } from 'react-router-dom'
import { Layout } from './components/Layout'
import { ChatPage } from './pages/ChatPage'
import { VoiceCallsList } from './pages/VoiceCallsList'
import { VoiceCallDetail } from './pages/VoiceCallDetail'

export default function App() {
  return (
    <Routes>
      <Route path="/" element={<Layout />}>
        <Route index element={<ChatPage />} />
        <Route path="calls" element={<VoiceCallsList />} />
        <Route path="calls/:callSid" element={<VoiceCallDetail />} />
      </Route>
    </Routes>
  )
}
