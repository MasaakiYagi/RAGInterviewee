from flask import Flask, render_template, request, jsonify, Response, session, make_response
import os
from dotenv import load_dotenv
from functools import wraps
from openai import OpenAI
import time
import io
import soundfile as sf
import base64

app = Flask(__name__)
app.secret_key = 'secret_key'

# Load API key and assistant ID from .env file
load_dotenv()
API_KEY = os.getenv('OPENAI_API_KEY')
ASSISTANT_ID = os.getenv('ASSISTANT_ID')
USERNAME = os.getenv('BASIC_AUTH_USERNAME', 'admin')
PASSWORD = os.getenv('BASIC_AUTH_PASSWORD', 'password')

class SingletonMeta(type):
    _instances = {}

    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            instance = super().__call__(*args, **kwargs)
            cls._instances[cls] = instance
        return cls._instances[cls]

class AIAssistant(metaclass=SingletonMeta):
    def __init__(self, assistant_id, api_key):
        self.assistant_id = assistant_id
        self.client = OpenAI(api_key=api_key)
        self.stt_model = "whisper-1"
        self.tts_model = "tts-1"
        self.voice_code = "nova"
        self.thread_id = self.client.beta.threads.create().id

    # ユーザー入力の文字起こし
    def transcribe_audio(self, audio_stream):
        # ファイルを読み込んでAPIに送信
        # with open(file_path, 'rb') as audio_file:
        #     transcript = self.client.audio.transcriptions.create(model=self.stt_model, file=audio_file)
        # バイトストリームをwav形式に変換
        data, samplerate = sf.read(audio_stream)
        wav_io = io.BytesIO()
        wav_io.name = "input.wav"
        sf.write(wav_io, data, samplerate, format='WAV')
        wav_io.seek(0)  # バッファの先頭に戻す

        transcript = self.client.audio.transcriptions.create(model=self.stt_model, file=wav_io)
        return transcript.text
    
    # LLMの応答生成
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

    # 応答音声の生成
    def text_to_speech(self, text):
        response = self.client.audio.speech.create(model=self.tts_model, voice=self.voice_code, input=text)
        byte_stream = io.BytesIO(response.content)
        # audio_data, samplerate = sf.read(byte_stream)
        # audio_file_path = "output.wav"
        # sf.write(audio_file_path, audio_data, samplerate)
        return byte_stream

    # 全てを順番に実行するラップ関数
    def reply_process(self, audio_stream):
        transcribed_text = self.transcribe_audio(audio_stream)
        reply_message = self.run_thread_actions(transcribed_text)
        audio_byte_stream = self.text_to_speech(reply_message)

        return transcribed_text, reply_message, audio_byte_stream


def check_auth(username, password):
    return username == USERNAME and password == PASSWORD

def authenticate():
    return Response(
        'Please provide valid credentials\n',
        401,
        {'WWW-Authenticate': 'Basic realm="Login Required"'}
    )

def requires_auth(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        auth = request.authorization
        if not auth or not check_auth(auth.username, auth.password):
            return authenticate()
        return f(*args, **kwargs)
    return decorated

assistant = AIAssistant(assistant_id=ASSISTANT_ID, api_key=API_KEY)

@app.before_request
def before_request():
    if 'status' not in session:
        session['status'] = '停止中'

@app.route('/')
@requires_auth
def index():
    session['status'] = '停止中'
    return render_template('index.html')

@app.route('/status', methods=['GET'])
def get_status():
    return jsonify(status=session.get('status', '停止中'))

@app.route('/start', methods=['POST'])
def start():
    if 'audio' not in request.files:
        return jsonify({'error': 'No audio file in request'}), 400

    audio_file = request.files['audio']
    if audio_file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    # audio_path = os.path.join('uploads', 'recording.wav')
    # audio_file.save(audio_path)
    # バイトストリームとして読み込む
    audio_stream = io.BytesIO(audio_file.read())

    # 応答生成
    # user_text, assistant_text, response_audio_path = assistant.reply_process(audio_path)
    user_text, assistant_text, response_audio_stream = assistant.reply_process(audio_stream)

    # バイトストリームをBase64に変換
    audio_data = response_audio_stream.getvalue()
    audio_base64 = base64.b64encode(audio_data).decode('utf-8')
    # with open(response_audio_path, 'rb') as f:
    #     audio_data = f.read()

    # audio_base64 = base64.b64encode(audio_data).decode('utf-8')

    return jsonify({
        'user': user_text,
        'assistant': assistant_text,
        'audio': audio_base64
    })

@app.route('/transcribe', methods=['POST'])
def transcribe():
    if 'audio' not in request.files:
        return jsonify({'error': 'No audio file in request'}), 400

    audio_file = request.files['audio']
    if audio_file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    # バイトストリームとして読み込む
    audio_stream = io.BytesIO(audio_file.read())

    # もじおこし
    user_text = assistant.transcribe_audio(audio_stream)

    return jsonify({
        'usertext': user_text,
    })

@app.route('/llm', methods=['POST'])
def llm():
    if not request.is_json:
        return jsonify({'error': 'Request must be JSON'}), 400

    data = request.get_json()

    if 'message' not in data:
        return jsonify({'error': 'No message in request'}), 400

    user_text = data['message']
    assistant_text = assistant.run_thread_actions(user_text)

    return jsonify({
        'assistanttext': assistant_text,
    })

@app.route('/tts', methods=['POST'])
def tts():
    if not request.is_json:
        return jsonify({'error': 'Request must be JSON'}), 400

    data = request.get_json()

    if 'message' not in data:
        return jsonify({'error': 'No message in request'}), 400

    assistant_text = data['message']
    response_audio_stream = assistant.text_to_speech(assistant_text)

    # バイトストリームをBase64に変換
    audio_data = response_audio_stream.getvalue()
    audio_base64 = base64.b64encode(audio_data).decode('utf-8')

    # audio_base64 = base64.b64encode(audio_data).decode('utf-8')

    return jsonify({
        'audio': audio_base64
    })

if __name__ == '__main__':
    if not os.path.exists('uploads'):
        os.makedirs('uploads')
    app.run(debug=True)
