import os
import asyncio
from flask import Flask, render_template, request, jsonify, session, Response
from openai import OpenAI
from dotenv import load_dotenv
import soundfile as sf
from functools import wraps
import threading
import time
import io

app = Flask(__name__)
app.secret_key = 'secret_key'

# Load API key and assistant ID from .env file
load_dotenv()
api_key = os.getenv('OPENAI_API_KEY')
assistant_id = os.getenv('ASSISTANT_ID')
USERNAME = os.getenv('BASIC_AUTH_USERNAME', 'admin')
PASSWORD = os.getenv('BASIC_AUTH_PASSWORD', 'password')

current_thread_id = None
thread_lock = threading.Lock()

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

    def start_thread(self):
        global current_thread_id
        with thread_lock:
            if current_thread_id is not None:
                self.stop_thread(current_thread_id)
            self.thread_id = self.client.beta.threads.create().id
            current_thread_id = self.thread_id

    def stop_thread(self, thread_id):
        # Here you can implement any cleanup if necessary
        pass

    def transcribe_audio(self, audio_bytes):
        # メモリ上の音声データをWAV形式に変換
        audio_file = io.BytesIO(audio_bytes)
        audio_data, samplerate = sf.read(audio_file, dtype='int16')
        wav_io = io.BytesIO()
        sf.write(wav_io, audio_data, samplerate, format='wav')
        wav_io.seek(0)
        
        transcript = self.client.audio.transcriptions.create(model=self.stt_model, file=wav_io)
        return transcript.text

    def run_thread_actions(self, text):
        self.client.beta.threads.messages.create(thread_id=self.thread_id, role="user", content=text)
        run = self.client.beta.threads.runs.create(thread_id=self.thread_id, assistant_id=self.assistant_id)
        while True:
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
        audio_file_path = "output.wav"
        sf.write(audio_file_path, audio_data, samplerate)
        return audio_file_path

    def interaction(self, audio_bytes):
        transcript_text = self.transcribe_audio(audio_bytes)
        assistant_content = self.run_thread_actions(transcript_text)
        audio_file_path = asyncio.run(self.text_to_speech(assistant_content))
        return transcript_text, assistant_content, audio_file_path

ai_assistant = AIAssistant(assistant_id=assistant_id, api_key=api_key)

@app.before_request
def before_request():
    if 'status' not in session:
        session['status'] = '停止中'
        ai_assistant.start_thread()

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
    audio_bytes = audio_file.read()
    session['status'] = 'ユーザー音声聞き取り中'
    transcript_text, assistant_content, audio_file_path = ai_assistant.interaction(audio_bytes)
    session['status'] = '停止中'
    return jsonify(user=transcript_text, assistant=assistant_content, audio_file=audio_file_path)

@app.route('/pause', methods=['POST'])
@requires_auth
def pause_interaction():
    session['status'] = '一時停止中'
    return jsonify(status='paused')

@app.route('/end', methods=['POST'])
@requires_auth
def end_interaction():
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
