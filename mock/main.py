import os
import streamlit as st
from openai import OpenAI
from dotenv import load_dotenv
import io
import time
import sounddevice as sd
import soundfile as sf

# .envファイルからAPIキーとアシスタントIDを読み込む
load_dotenv()
api_key = os.getenv('OPENAI_API_KEY')
assistant_id = os.getenv('ASSISTANT_ID')

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

    def __init__(self, assistant_id: str, api_key: str):
        """
        初期化処理。

        Args:
            assistant_id (str): AIアシスタントのID。
        """
        self.assistant_id = assistant_id
        self.client = OpenAI()
        self.client.api_key = api_key
        thread = self.client.beta.threads.create()
        self.thread_id = thread.id

    def record_audio(self) -> any:
        """
        オーディオを録音する。

        Returns:
            any: 録音データ。
        """
        st.write("Start recording...")
        recorded_data = sd.rec(
            int(self.duration * self.fs), samplerate=self.fs, channels=self.channels
        )
        sd.wait()  # 録音が終わるまで待機
        st.write("...Finished recording")
        return recorded_data

    def transcribe_audio(self, audio_data: any) -> str:
        """
        録音されたオーディオをテキストに変換する。

        Args:
            audio_data (any): 録音データ。

        Returns:
            str: 変換されたテキスト。
        """
        audio_file = io.BytesIO()
        sf.write(audio_file, audio_data, self.fs, format='wav')
        audio_file.seek(0)
        
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
        audio_data, samplerate = sf.read(byte_stream)

        sd.play(audio_data, samplerate)
        sd.wait()

def main():
    ai_assistant = AIAssistant(assistant_id=assistant_id, api_key=api_key)

    st.title("AI Assistant with Speech-to-Text and Text-to-Speech")
    
    # キャラクター画像を表示
    st.image("character.png")

    # コンテナを追加
    container = st.container()

    if st.button("Record Audio"):
        while True:
            recorded_data = ai_assistant.record_audio()
            transcript_text = ai_assistant.transcribe_audio(recorded_data)
            container.write(f"user: {transcript_text}")

            if transcript_text:
                assistant_content = ai_assistant.run_thread_actions(transcript_text)
                container.write(f"assistant: {assistant_content}")
                ai_assistant.text_to_speech(assistant_content)

if __name__ == "__main__":
    main()
