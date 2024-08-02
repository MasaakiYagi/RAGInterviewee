import os
from quart import Quart, render_template, request, jsonify, session
from openai import OpenAI
from dotenv import load_dotenv
import io
import asyncio
import sounddevice as sd
from scipy.io.wavfile import write

app = Quart(__name__)
app.secret_key = 'secret_key'

# Load API key and assistant ID from .env file
load_dotenv()
api_key = os.getenv('OPENAI_API_KEY')
assistant_id = os.getenv('ASSISTANT_ID')

class AIAssistant:
    def __init__(self, assistant_id, api_key):
        self.assistant_id = assistant_id
        self.client = OpenAI(api_key=api_key)
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
        session['status'] = 'ユーザー音声聞き取り中'
        print("Start recording...")
        recording = sd.rec(int(self.duration * self.fs), samplerate=self.fs, channels=self.channels)
        sd.wait()
        print("Finished recording")
        return recording

    async def transcribe_audio(self, audio_data):
        session['status'] = 'AI応答中'
        write("temp.wav", self.fs, audio_data)
        with open("temp.wav", "rb") as audio_file:
            transcript = self.client.audio.transcriptions.create(model=self.stt_model, file=audio_file)
        return transcript.text

    async def run_thread_actions(self, text):
        self.client.beta.threads.messages.create(thread_id=self.thread_id, role="user", content=text)
        run = self.client.beta.threads.runs.create(thread_id=self.thread_id, assistant_id=self.assistant_id)
        while True:
            result = self.client.beta.threads.runs.retrieve(thread_id=self.thread_id, run_id=run.id)
            if result.status == 'completed':
                break
            await asyncio.sleep(0.5)
        messages = self.client.beta.threads.messages.list(thread_id=self.thread_id, order='asc')
        if len(messages.data) < 2:
            return ""
        return messages.data[-1].content[0].text.value

    async def interaction(self):
        recorded_data = await self.record_audio()
        transcript_text = await self.transcribe_audio(recorded_data)
        assistant_content = await self.run_thread_actions(transcript_text)
        return transcript_text, assistant_content

ai_assistant = AIAssistant(assistant_id=assistant_id, api_key=api_key)

@app.route('/')
async def index():
    if 'status' not in session:
        session['status'] = '停止中'
    return await render_template('index.html', status=session['status'])

@app.route('/start', methods=['POST'])
async def start_interaction():
    await ai_assistant.start_thread()
    ai_assistant.is_running = True
    session['status'] = 'ユーザー音声聞き取り中'
    return jsonify(status='started')

@app.route('/pause', methods=['POST'])
async def pause_interaction():
    ai_assistant.is_running = False
    session['status'] = '一時停止中'
    return jsonify(status='paused')

@app.route('/end', methods=['POST'])
async def end_interaction():
    ai_assistant.is_running = False
    session['status'] = '停止中'
    return jsonify(status='ended')

@app.route('/interact', methods=['POST'])
async def interact():
    if ai_assistant.is_running:
        transcript_text, assistant_content = await ai_assistant.interaction()
        return jsonify(user=transcript_text, assistant=assistant_content)
    return jsonify(status='not running')

@app.route('/status', methods=['GET'])
async def get_status():
    return jsonify(status=session.get('status', '停止中'))

if __name__ == '__main__':
    app.run(debug=True)
