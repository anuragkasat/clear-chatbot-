import streamlit as st

def setup_ui():
    st.set_page_config(page_title="CASme Chatbot", page_icon="✨", layout="wide")

    # Apply custom styling
    st.markdown(
        """
        <style>
            body { background-color: white; }
            .main { background-color: white; }
            .stTextInput>div>div>input {
                border-radius: 10px;
                border: 1px solid #ccc;
                padding: 10px;
            }
            .stButton>button {
                background-color: #0078D4;
                color: white;
                border-radius: 10px;
                padding: 10px 20px;
            }
            .chat-container {
                background-color: white;
                border-radius: 10px;
                padding: 15px;
                box-shadow: 0px 4px 6px rgba(0, 0, 0, 0.1);
            }
            .chat-bubble {
                background-color: #f1f1f1;
                padding: 10px;
                border-radius: 10px;
                margin-bottom: 10px;
            }
        </style>
        """,
        unsafe_allow_html=True
    )

    # Sidebar with small logo
    st.sidebar.image("clear.jpeg", width=120)
    hf_api_key = st.sidebar.text_input("🔑 Enter Hugging Face API Key", type="password")

    # Title and description
    st.image("clear.jpeg", width=90)
    st.title("✨AI Assisted CASme ChatBot")
    st.write("Chat Now!")

    return hf_api_key
