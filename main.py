import os
from openai import OpenAI
from dotenv import load_dotenv
import io
import time
import sounddevice as sd
from scipy.io.wavfile import write
from pydub import AudioSegment
from pydub.playback import play

# .envファイルからAPIキーとアシスタントIDを読み込む
load_dotenv()
api_key = os.getenv('OPENAI_API_KEY')
assistant_id = os.getenv('ASSISTANT_ID')

# # クライアント設定
# client = OpenAI()
# client.api_key = api_key

# STT, Assistants, TTSのパイプラインクラス
class AIAssistant:
    """
    OpenAIのAPIを利用して音声をテキストに変換し、AIアシスタントで処理し、音声に戻すクラス。
    """

    # 録音パラメータ
    fs = 44100  # サンプリングレート
    duration = 5  # 録音する秒数
    channels = 1  # モノラル録音
    # 音声認識モデル
    stt_model = "whisper-1"
    # 音声生成モデル
    tts_model = "tts-1"  # 高品質モデル tts-1-hd
    # 声質
    voice_code = "nova"  # 男性 alloy, echo, fable, onyx 女性 nova, shimmer

    def __init__(self, assistant_id: str, api_key: str, output_audio_file: str = "./output.wav"):
        """
        初期化処理。

        Args:
            assistant_id (str): AIアシスタントのID。
            output_audio_file (str, optional): 音声ファイルの保存先。
        """
        self.assistant_id = assistant_id
        self.client = OpenAI()
        self.client.api_key = api_key
        thread = self.client.beta.threads.create()
        self.thread_id = thread.id
        self.output_audio_file = output_audio_file

    def record_audio(self) -> any:
        """
        オーディオを録音する。

        Returns:
            any: 録音データ。
        """
        print("Start recording...")
        recorded_data = sd.rec(
            int(self.duration * self.fs), samplerate=self.fs, channels=self.channels
        )
        sd.wait()  # 録音が終わるまで待機
        print("...Finished recording")
        return recorded_data

    def transcribe_audio(self, audio_data: any) -> str:
        """
        録音されたオーディオをテキストに変換する。

        Args:
            audio_data (any): 録音データ。

        Returns:
            str: 変換されたテキスト。
        """
        write(self.output_audio_file, self.fs, audio_data)
        with open(self.output_audio_file, "rb") as audio_file:
            transcript = self.client.audio.transcriptions.create(
                model=self.stt_model, file=audio_file
            )
        return transcript.text

    def run_thread_actions(self, text: str) -> str:
        """
        テキストをAIアシスタントに送信し、応答を取得する。

        Args:
            text (str): ユーザーからのテキスト。

        Returns:
            str: アシスタントからの応答テキスト。
        """
        self.client.beta.threads.messages.create(
            thread_id=self.thread_id,
            role="user",
            content=text,
        )

        run = self.client.beta.threads.runs.create(
            thread_id=self.thread_id,
            assistant_id=self.assistant_id,
        )

        while True:
            result = self.client.beta.threads.runs.retrieve(
                thread_id=self.thread_id, run_id=run.id
            )
            if result.status == "completed":
                break
            time.sleep(0.5)

        messages = self.client.beta.threads.messages.list(
            thread_id=self.thread_id, order="asc"
        )

        if len(messages.data) < 2:
            return ""

        return messages.data[-1].content[0].text.value

    def text_to_speech(self, text: str) -> None:
        """
        テキストを音声に変換し、再生する。

        Args:
            text (str): 再生するテキスト。
        """
        response = self.client.audio.speech.create(
            model=self.tts_model,
            voice=self.voice_code,
            input=text,
        )

        byte_stream = io.BytesIO(response.content)

        audio = AudioSegment.from_file(byte_stream, format="mp3")

        play(audio)

def main():
    ai_assistant = AIAssistant(assistant_id=assistant_id)

    while True:
        recorded_data = ai_assistant.record_audio()
        transcript_text = ai_assistant.transcribe_audio(recorded_data)
        print(f"user: {transcript_text}")

        if not transcript_text:
            break

        assistant_content = ai_assistant.run_thread_actions(transcript_text)
        print(f"assistant: {assistant_content}")
        ai_assistant.text_to_speech(assistant_content)

