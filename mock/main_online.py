import os
import streamlit as st
from openai import OpenAI
from dotenv import load_dotenv
import io
import time
import asyncio
import sounddevice as sd
import soundfile as sf
from scipy.io.wavfile import write

# .envファイルからAPIキーとアシスタントIDを読み込む
load_dotenv()
api_key = os.getenv('OPENAI_API_KEY')
assistant_id = os.getenv('ASSISTANT_ID')

class AIAssistant:
    def __init__(self, assistant_id: str, api_key: str):
        self.assistant_id = assistant_id
        self.client = OpenAI()
        self.client.api_key = api_key
        self.stt_model = "whisper-1"
        self.tts_model = "tts-1"
        self.voice_code = "nova"
        self.fs = 44100
        self.duration = 5
        self.channels = 1
        self.thread_id = None
        self.is_running = False

    async def start_thread(self):
        self.thread_id = self.client.beta.threads.create().id

    async def record_audio(self):
        print("Start recording...")
        loop = asyncio.get_event_loop()
        recording = await loop.run_in_executor(None, lambda: sd.rec(int(self.duration * self.fs), samplerate=self.fs, channels=self.channels))
        sd.wait()
        print("...Finished recording")
        return recording

    async def transcribe_audio(self, audio_data):
        write("temp.wav", self.fs, audio_data)
        with open("temp.wav", "rb") as audio_file:
            transcript = self.client.audio.transcriptions.create(model=self.stt_model, file=audio_file)
        return transcript.text

    async def run_thread_actions(self, text):
        self.client.beta.threads.messages.create(thread_id=self.thread_id, role="user", content=text)
        run = self.client.beta.threads.runs.create(thread_id=self.thread_id, assistant_id=self.assistant_id)
        while True:
            result = self.client.beta.threads.runs.retrieve(thread_id=self.thread_id, run_id=run.id)
            if result.status == "completed":
                break
            await asyncio.sleep(0.5)
        messages = self.client.beta.threads.messages.list(thread_id=self.thread_id, order="asc")
        if len(messages.data) < 2:
            return ""
        return messages.data[-1].content[0].text.value

    async def text_to_speech(self, text):
        response = self.client.audio.speech.create(model=self.tts_model, voice=self.voice_code, input=text)
        byte_stream = io.BytesIO(response.content)
        audio_data, samplerate = sf.read(byte_stream)
        sd.play(audio_data, samplerate)
        sd.wait()

    async def interaction_loop(self):
        self.is_running = True
        while self.is_running:
            try:
                recorded_data = await self.record_audio()
                transcript_text = await self.transcribe_audio(recorded_data)
                yield ("user", transcript_text)
                if transcript_text:
                    assistant_content = await self.run_thread_actions(transcript_text)
                    await self.text_to_speech(assistant_content)
                    yield ("assistant", assistant_content)
            except Exception as e:
                self.is_running = False
                raise e

    def stop_interaction(self):
        self.is_running = False

async def main():
    if 'is_running' not in st.session_state:
        st.session_state.is_running = False
    if 'messages' not in st.session_state:
        st.session_state.messages = []

    ai_assistant = AIAssistant(assistant_id=assistant_id, api_key=api_key)

    st.title("AI Assistant with Real-Time Speech-to-Text and Text-to-Speech")

    # カスタムCSSを追加してスクロール可能な枠を作成
    st.markdown(
        """
        <style>
        .scrollable-container {
            height: 400px;
            overflow-y: scroll;
            border: 1px solid #ccc;
            padding: 10px;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    if st.button("Start Interaction"):
        st.session_state.is_running = True
        await ai_assistant.start_thread()

    if st.button("Pause Interaction"):
        ai_assistant.stop_interaction()
        st.session_state.is_running = False

    if st.button("End Interaction"):
        ai_assistant.stop_interaction()
        st.session_state.is_running = False
        st.write("Interaction ended.")

    # スクロール可能な枠にテキストを表示
    st.markdown('<div class="scrollable-container">', unsafe_allow_html=True)
    for role, message in st.session_state.messages:
        if role == "user":
            st.write(f"user: {message}")
        elif role == "assistant":
            st.write(f"assistant: {message}")
    st.markdown('</div>', unsafe_allow_html=True)

    if st.session_state.is_running:
        interaction_generator = ai_assistant.interaction_loop()
        try:
            while True:
                role, content = await interaction_generator.__anext__()
                st.session_state.messages.append((role, content))
        except StopAsyncIteration:
            pass
        except Exception as e:
            st.error(f"An error occurred: {e}")

if __name__ == "__main__":
    asyncio.run(main())
