import streamlit as st
from langchain_groq import ChatGroq
from langchain_core.messages import AIMessage, HumanMessage

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="DEBUG - Groq API Test",
    page_icon="🐛",
    layout="wide"
)

st.title("🐛 Groq API Direct Test")
st.markdown("This is a simple test to check if the Groq API key and connection are working.")

# --- Initialize Chat History ---
if "chat_history" not in st.session_state:
    st.session_state.chat_history = [AIMessage(content="Hello! This is a direct test of the Groq LLM. Ask me anything simple.")]

# --- API KEY & MAIN LOGIC ---
groq_api_key = st.secrets.get("GROQ_API_KEY")

if not groq_api_key:
    st.info("Please add your Groq API Key to the Streamlit secrets to run this app.")
    st.stop()

# --- Basic LLM Chain ---
try:
    llm = ChatGroq(model="llama3-8b-8192", groq_api_key=groq_api_key, temperature=0.7)
    
    # Display chat history
    for message in st.session_state.chat_history:
        if isinstance(message, AIMessage):
            with st.chat_message("AI"):
                st.write(message.content)
        elif isinstance(message, HumanMessage):
            with st.chat_message("Human"):
                st.write(message.content)

    # Get user input
    user_query = st.chat_input("Ask a simple question...")
    if user_query:
        st.session_state.chat_history.append(HumanMessage(content=user_query))
        with st.chat_message("Human"):
            st.write(user_query)

        with st.chat_message("AI"):
            with st.spinner("LLM is thinking..."):
                # Pass the history to the LLM
                response = llm.invoke(st.session_state.chat_history)
                answer = response.content
                st.session_state.chat_history.append(AIMessage(content=answer))
                st.rerun()

except Exception as e:
    st.error(f"An error occurred while initializing or calling the LLM: {e}")
