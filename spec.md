Voice conversation app.
** Description: An app for testing voice integrations using different ASR models and different AI systems (OpenAI and Amelia)


■ Frontend: Traditional chat interface with a "start voice conversation" feature
- Tech stack: React, Typescript, Vite, Axios, Material-UI, i18next
- Thin left section that runs the full vertical length of the screen
- Domain selection box in the left of the screen
- Integration selection box in the left screen: Choices: 1) OpenAI 2) Amelia
- There should be a toggle called: "continuous voice mode"
- Language selection also in the left part of the screen (choices: EN and JA)
- Voice model selection also in the left part of the screen (stub out a couple selections like WhisperV2, WhisperV3, and Google. I'll update it the coverage)
- Chat facilities are on the remaining right side of the screen
- Chat input area at the bottom of the right side of the screen.
- Chat input area should have a button on the right that reads: "start voice conversation"
- Voice input should be "auto-detected" for both start and stop. (Created a voice_input_design.md doc describing the design/implementation.)
- Most of the right side of the screen should be the chat history.
- chat history is: agent on the left and user on the right
- Japanese language should be supported throughout

■ Backend
- Tech stack: FastAPI, uv (no pip), OpenAI
- (I've pulled over a few items from another project that starts)
- Phase 1: Get Google ASR and TTS working
- I have an ENV file in project root with OPENAI_API_KEY, GOOGLE_PROJECT, and GOOGLE_APPLICATION_CREDENTIALS specified
- The GOOGLE_APPLICATION_CREDENTIALS file is also in root for now.

■ Integration with Twilio
- Phase 2
■ Integration with Amelia
- Phase 3 

■ Operations
- Make all operations available using "Tasks" (Taskfile.yaml)