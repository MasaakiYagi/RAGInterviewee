import os
import streamlit as st
from openai import OpenAI
from dotenv import load_dotenv
import io
import time
import sounddevice as sd
import soundfile as sf
from scipy.io.wavfile import write

# .envファイルからAPIキーとアシスタントIDを読み込む
load_dotenv()
api_key = os.getenv('OPENAI_API_KEY')
assistant_id = os.getenv('ASSISTANT_ID')

def main():
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

    # セッション状態にメッセージリストを保持
    if 'messages' not in st.session_state:
        st.session_state.messages = []

    # コンテナを作成して、その中にテキストを追加していく
    container = st.markdown('<div class="scrollable-container"></div>', unsafe_allow_html=True)

    # ボタンを押したときにメッセージを追加
    if st.button("Add Message"):
        st.session_state.messages.append("New message added!")

    # メッセージリストをコンテナに表示
    container.markdown('<div class="scrollable-container">', unsafe_allow_html=True)
    for message in st.session_state.messages:
        container.markdown(f"<p>{message}</p>", unsafe_allow_html=True)
    container.markdown('</div>', unsafe_allow_html=True)

if __name__ == "__main__":
    main()
