/**
 * Strip markdown formatting so TTS reads plain text (no "asterisk asterisk" etc).
 */
export function stripMarkdownForTTS(text: string): string {
  if (!text?.trim()) return text ?? ''
  let out = text
  // Code blocks (remove entire block, keep content as plain text)
  out = out.replace(/```[\s\S]*?```/g, ' ')
  // Inline code
  out = out.replace(/`[^`]+`/g, (m) => m.slice(1, -1))
  // Bold/italic: **x** __x__ *x* _x_
  out = out.replace(/\*\*([^*]+)\*\*/g, '$1')
  out = out.replace(/__([^_]+)__/g, '$1')
  out = out.replace(/\*([^*]+)\*/g, '$1')
  out = out.replace(/_([^_]+)_/g, '$1')
  // Headers: # ## ### etc
  out = out.replace(/^#{1,6}\s+/gm, '')
  // Strikethrough
  out = out.replace(/~~([^~]+)~~/g, '$1')
  // Links: [text](url) -> text
  out = out.replace(/\[([^\]]+)\]\([^)]+\)/g, '$1')
  // List markers
  out = out.replace(/^\s*[-*+]\s+/gm, ' ')
  out = out.replace(/^\s*\d+\.\s+/gm, ' ')
  // Collapse multiple spaces/newlines
  out = out.replace(/\n{2,}/g, '\n').replace(/[ \t]+/g, ' ').trim()
  return out
}
