import os
import streamlit as st
import pandas as pd
from prophet import Prophet
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
# --- CHANGE 1: Import ChatNVIDIA instead of ChatGroq ---
from langchain_nvidia_ai_endpoints import ChatNVIDIA
from langchain.prompts import ChatPromptTemplate
from langchain_community.utilities import SQLDatabase
# --- REMOVED create_retriever_tool as we are making a custom tool ---
from langchain_community.agent_toolkits import create_sql_agent
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain_core.messages import AIMessage, HumanMessage
from langchain.tools import Tool
from langchain_core.prompts import MessagesPlaceholder
# --- REMOVED groq.BadRequestError as it's no longer needed ---

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="India Economic Data Navigator",
    page_icon="🇮🇳",
    layout="wide"
)

# --- CACHED FUNCTIONS ---
@st.cache_resource
def load_embedding_model():
    model_name = "sentence-transformers/all-MiniLM-L6-v2"
    model_kwargs = {'device': 'cpu'}
    return HuggingFaceEmbeddings(model_name=model_name, model_kwargs=model_kwargs)

@st.cache_resource
def load_pdf_retriever(_embeddings):
    # Ensure the 'vector_store' directory exists
    if not os.path.exists('vector_store'):
        st.error("The 'vector_store' directory was not found. Please make sure it exists and contains your FAISS index.")
        st.stop()
    persist_directory = 'vector_store'
    vectordb = FAISS.load_local(
        folder_path=persist_directory,
        embeddings=_embeddings,
        allow_dangerous_deserialization=True
    )
    return vectordb.as_retriever(search_kwargs={"k": 5})

@st.cache_resource
def get_sql_database():
    db_path = 'data/processed/esd_indicators.sqlite'
    # Ensure the database file exists
    if not os.path.exists(db_path):
        st.error(f"The database file was not found at '{db_path}'. Please ensure the path is correct.")
        st.stop()
    db_uri = f"sqlite:///{db_path}"
    return SQLDatabase.from_uri(db_uri, include_tables=['indicators', 'gsec_auctions'])

@st.cache_data
def get_gdp_forecast(state_name: str, years_to_forecast: int = 3):
    try:
        df = pd.read_csv('data/processed/final_forecasting_dataset.csv')
        state_df = df[df['State'].str.contains(state_name, case=False, na=False)].copy()

        if state_df.empty:
            return f"Could not find data for the state: {state_name}"

        state_df.rename(columns={'Year': 'ds', 'GSDP': 'y'}, inplace=True)
        state_df['ds'] = pd.to_datetime(state_df['ds'], format='%Y')

        regressors = ['Fiscal_Deficit_Percent', 'CPI_Inflation', 'Agri_Production_Thousand_Tonnes']
        state_df.ffill(inplace=True).bfill(inplace=True)

        if not all(col in state_df.columns for col in regressors):
            return "Not all required data columns are available for this state to make a forecast."

        model = Prophet(yearly_seasonality=True)
        for reg in regressors:
            model.add_regressor(reg)

        model.fit(state_df)

        future = model.make_future_dataframe(periods=years_to_forecast, freq='Y')
        for reg in regressors:
            future[reg] = state_df[reg].iloc[-1]

        forecast = model.predict(future)

        forecast_summary = f"GSDP Forecast for {state_name.title()} (in ₹ crore):\n"
        for _, row in forecast.tail(years_to_forecast).iterrows():
            year = row['ds'].year
            predicted_gsdp = f"{row['yhat']:,.0f}"
            forecast_summary += f"- {year}: ₹{predicted_gsdp}\n"

        return forecast_summary
    except FileNotFoundError:
        return "Error: The forecasting dataset 'data/processed/final_forecasting_dataset.csv' was not found."
    except Exception as e:
        return f"An error occurred during forecasting: {e}"

# --- FIX: ROBUST PDF SEARCH FUNCTION TO PREVENT ERRORS ---
def pdf_search(user_query: str) -> str:
    """
    Searches the vector store, combines the content of retrieved documents into a
    single string, and truncates it to prevent API context length errors.
    """
    try:
        retrieved_docs = pdf_retriever.invoke(user_query)
        # Combine the content of all retrieved documents
        combined_content = "\n---\n".join([doc.page_content for doc in retrieved_docs])
        # Truncate the combined content to a safe limit
        return combined_content[:6000] # Increased limit slightly for potentially better context
    except Exception as e:
        return f"An error occurred during PDF search: {e}"

# --- STREAMLIT APP LAYOUT ---
st.title("🇮🇳 India Economic Data Navigator")
st.markdown("I can answer questions, search documents, and **forecast GSDP**.")

if "chat_history" not in st.session_state:
    st.session_state.chat_history = [AIMessage(content="Hello! How can I help you?")]

# --- CHANGE 2: USE NVIDIA API KEY ---
nvidia_api_key = st.secrets.get("NVIDIA_API_KEY")

if not nvidia_api_key:
    st.info("Please add your NVIDIA API Key to the Streamlit secrets to run this app.")
    st.stop()

# Initialize components
# --- CHANGE 3: INITIALIZE ChatNVIDIA LLM ---
llm = ChatNVIDIA(model="meta/llama3-70b-instruct", nvidia_api_key=nvidia_api_key, temperature=0, max_tokens=1024)
embeddings = load_embedding_model()
pdf_retriever = load_pdf_retriever(embeddings)
db = get_sql_database()

# --- FIX: DEFINE TOOLS USING THE ROBUST CUSTOM SEARCH FUNCTION ---
pdf_search_tool = Tool(
    name="economic_data_search",
    func=pdf_search, # Use the robust function
    description="Use for questions about India's economy, policies, and analyses from official reports. Input should be the user's question."
)

sql_agent_executor = create_sql_agent(llm, db=db, agent_type="openai-tools", verbose=False)
sql_tool = Tool(
    name="database_search",
    func=sql_agent_executor.invoke,
    description="Use for questions about specific numbers of business establishments or G-Sec auctions."
)

forecasting_tool = Tool(
    name="gsdp_forecaster",
    func=get_gdp_forecast,
    description="Use this to forecast future GSDP for a specific Indian state."
)

tools = [pdf_search_tool, sql_tool, forecasting_tool]

# --- CREATE THE AGENT ---
agent_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You are an expert financial assistant. You have access to tools to answer questions. For forecasts, mention that they are based on historical data and not financial advice."),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{input}"),
        ("placeholder", "{agent_scratchpad}"),
    ]
)
agent = create_tool_calling_agent(llm, tools, agent_prompt)
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True, handle_parsing_errors=True)

# Display chat history
for message in st.session_state.chat_history:
    if isinstance(message, AIMessage):
        with st.chat_message("AI"):
            st.write(message.content)
    elif isinstance(message, HumanMessage):
        with st.chat_message("Human"):
            st.write(message.content)

# Get user input
user_query = st.chat_input("Ask your question...")
if user_query:
    st.session_state.chat_history.append(HumanMessage(content=user_query))
    with st.chat_message("Human"):
        st.write(user_query)

    with st.chat_message("AI"):
        with st.spinner("Agent is thinking..."):
            try:
                # Limit the chat history to keep the prompt size manageable
                recent_history = st.session_state.chat_history[-4:]

                response = agent_executor.invoke({
                    "input": user_query,
                    "chat_history": recent_history
                })
                answer = response.get("output", "I encountered an error.")
            # --- CHANGE 4: GENERAL EXCEPTION HANDLING ---
            except Exception as e:
                st.error(f"An error occurred: {e}")
                answer = "I'm sorry, I ran into a problem while trying to answer. Please try again or rephrase your question."

        st.write(answer)
        st.session_state.chat_history.append(AIMessage(content=answer))
        # No st.rerun() needed, Streamlit handles the update
