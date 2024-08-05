import os
import asyncio
from flask import Flask, render_template, request, jsonify, session, Response, send_file
from openai import OpenAI
from dotenv import load_dotenv
import soundfile as sf
from functools import wraps
import threading
import time
import uuid
import io

app = Flask(__name__)
app.secret_key = 'secret_key'

# Load API key and assistant ID from .env file
load_dotenv()
api_key = os.getenv('OPENAI_API_KEY')
assistant_id = os.getenv('ASSISTANT_ID')

USERNAME = os.getenv('BASIC_AUTH_USERNAME', 'admin')
PASSWORD = os.getenv('BASIC_AUTH_PASSWORD', 'password')

stop_event = threading.Event()
interaction_thread = None

def check_auth(username, password):
    """ 認証情報をチェックする関数 """
    return username == USERNAME and password == PASSWORD

def authenticate():
    """ 認証が必要な場合に401レスポンスを返す関数 """
    return Response(
        'Please provide valid credentials\n',
        401,
        {'WWW-Authenticate': 'Basic realm="Login Required"'}
    )

def requires_auth(f):
    """ 認証を要求するデコレータ """
    @wraps(f)
    def decorated(*args, **kwargs):
        auth = request.authorization
        if not auth or not check_auth(auth.username, auth.password):
            return authenticate()
        return f(*args, **kwargs)
    return decorated

class AIAssistant:
    def __init__(self, assistant_id, api_key):
        self.assistant_id = assistant_id
        self.client = OpenAI(api_key=api_key)
        self.stt_model = "whisper-1"
        self.tts_model = "tts-1"
        self.voice_code = "nova"
        self.thread_id = None
        self.is_running = False

    def start_thread(self):
        self.thread_id = self.client.beta.threads.create().id

    def transcribe_audio(self, audio_path):
        if stop_event.is_set():
            raise InterruptedError("Transcription stopped")
        with open(audio_path, "rb") as audio_file:
            transcript = self.client.audio.transcriptions.create(model=self.stt_model, file=audio_file)
        return transcript.text

    def run_thread_actions(self, text):
        if stop_event.is_set():
            raise InterruptedError("Thread actions stopped")
        self.client.beta.threads.messages.create(thread_id=self.thread_id, role="user", content=text)
        run = self.client.beta.threads.runs.create(thread_id=self.thread_id, assistant_id=self.assistant_id)
        while True:
            if stop_event.is_set():
                raise InterruptedError("Thread actions stopped")
            result = self.client.beta.threads.runs.retrieve(thread_id=self.thread_id, run_id=run.id)
            if result.status == 'completed':
                break
            time.sleep(0.5)
        messages = self.client.beta.threads.messages.list(thread_id=self.thread_id, order='asc')
        if len(messages.data) < 2:
            return ""
        return messages.data[-1].content[0].text.value

    async def text_to_speech(self, text):
        response = self.client.audio.speech.create(model=self.tts_model, voice=self.voice_code, input=text)
        byte_stream = io.BytesIO(response.content)
        audio_data, samplerate = sf.read(byte_stream)
        audio_file_path = f"{uuid.uuid4()}.wav"
        sf.write(audio_file_path, audio_data, samplerate)
        return audio_file_path

    def interaction(self, audio_path):
        try:
            transcript_text = self.transcribe_audio(audio_path)
            assistant_content = self.run_thread_actions(transcript_text)
            audio_file_path = asyncio.run(self.text_to_speech(assistant_content))
            return transcript_text, assistant_content, audio_file_path
        except InterruptedError:
            return None, None, None

    def start_interaction(self, audio_path):
        global stop_event
        stop_event.clear()
        self.is_running = True
        transcript_text, assistant_content, audio_file_path = self.interaction(audio_path)
        if transcript_text is not None and assistant_content is not None:
            print(f"User: {transcript_text}")
            print(f"Assistant: {assistant_content}")
        return transcript_text, assistant_content, audio_file_path

    def stop_interaction(self):
        global stop_event
        self.is_running = False
        stop_event.set()

ai_assistant = AIAssistant(assistant_id=assistant_id, api_key=api_key)

@app.before_request
def before_request():
    session['status'] = '停止中'

@app.route('/')
@requires_auth
def index():
    return render_template('index.html', status=session['status'])

@app.route('/start', methods=['POST'])
@requires_auth
def start_interaction():
    if 'audio' not in request.files:
        return jsonify(status='error', message='Audio file is missing')
    audio_file = request.files['audio']
    audio_path = f"temp_{uuid.uuid4()}.wav"
    audio_file.save(audio_path)
    ai_assistant.start_thread()
    session['status'] = 'ユーザー音声聞き取り中'
    transcript_text, assistant_content, audio_file_path = ai_assistant.start_interaction(audio_path)
    session['status'] = '停止中'
    return jsonify(user=transcript_text, assistant=assistant_content, audio_file=audio_file_path)

@app.route('/pause', methods=['POST'])
@requires_auth
def pause_interaction():
    ai_assistant.is_running = False
    session['status'] = '一時停止中'
    return jsonify(status='paused')

@app.route('/end', methods=['POST'])
@requires_auth
def end_interaction():
    ai_assistant.stop_interaction()
    session['status'] = '停止中'
    return jsonify(status='ended')

@app.route('/audio/<filename>')
@requires_auth
def get_audio(filename):
    return send_file(filename)

@app.route('/status', methods=['GET'])
@requires_auth
def get_status():
    return jsonify(status=session.get('status', '停止中'))

if __name__ == '__main__':
    app.run(debug=True)
