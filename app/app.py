import os
import streamlit as st
import pandas as pd
from prophet import Prophet
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain.prompts import ChatPromptTemplate
from langchain_community.utilities import SQLDatabase
from langchain.tools.retriever import create_retriever_tool
from langchain_community.agent_toolkits import create_sql_agent
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain_core.messages import AIMessage, HumanMessage
from langchain.tools import Tool

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
    persist_directory = './vector_store'
    vectordb = Chroma(persist_directory=persist_directory, embedding_function=_embeddings)
    return vectordb.as_retriever(search_kwargs={"k": 5})

@st.cache_resource
def get_sql_database():
    db_uri = "sqlite:///./data/processed/esd_indicators.sqlite"
    return SQLDatabase.from_uri(db_uri, include_tables=['indicators', 'gsec_auctions'])

# --- NEW: FORECASTING FUNCTION ---
@st.cache_data
def get_gdp_forecast(state_name: str, years_to_forecast: int = 3):
    """
    Trains a Prophet model and returns a GSDP forecast for a specific state.
    """
    try:
        df = pd.read_csv('./data/processed/final_forecasting_dataset.csv')
        state_df = df[df['State'].str.contains(state_name, case=False, na=False)].copy()

        if state_df.empty:
            return f"Could not find data for the state: {state_name}"

        state_df.rename(columns={'Year': 'ds', 'GSDP': 'y'}, inplace=True)
        state_df['ds'] = pd.to_datetime(state_df['ds'], format='%Y')
        
        # Define regressors and fill missing values
        regressors = ['Fiscal_Deficit_Percent', 'CPI_Inflation', 'Agri_Production_Thousand_Tonnes']
        state_df.ffill(inplace=True).bfill(inplace=True)

        # Check if all regressor columns exist
        if not all(col in state_df.columns for col in regressors):
             return "Not all required data columns (regressors) are available for this state to make a forecast."

        model = Prophet(yearly_seasonality=True)
        for reg in regressors:
            model.add_regressor(reg)
        
        model.fit(state_df)
        
        future = model.make_future_dataframe(periods=years_to_forecast, freq='Y')
        for reg in regressors:
            future[reg] = state_df[reg].iloc[-1]
            
        forecast = model.predict(future)
        
        # Format the forecast into a readable string
        forecast_summary = f"GSDP Forecast for {state_name.title()} (in ₹ crore):\n"
        for _, row in forecast.tail(years_to_forecast).iterrows():
            year = row['ds'].year
            predicted_gsdp = f"{row['yhat']:,.0f}"
            forecast_summary += f"- {year}: ₹{predicted_gsdp}\n"
            
        return forecast_summary
    except Exception as e:
        return f"An error occurred during forecasting: {e}"


# --- STREAMLIT APP LAYOUT ---
st.title("🇮🇳 India Economic Data Navigator")
st.markdown("I can answer questions, search documents, and **forecast GSDP**.")

if "chat_history" not in st.session_state:
    st.session_state.chat_history = [AIMessage(content="Hello! I am an AI agent with access to multiple data sources and a forecasting model. How can I help you?")]

# --- API KEY ---
groq_api_key = st.sidebar.text_input("Enter your Groq API Key:", type="password")

# --- MAIN LOGIC ---
if groq_api_key:
    llm = ChatGroq(model="llama3-70b-8192", groq_api_key=groq_api_key, temperature=0)
    
    embeddings = load_embedding_model()
    pdf_retriever = load_pdf_retriever(embeddings)
    db = get_sql_database()

    # --- DEFINE TOOLS ---
    pdf_search_tool = create_retriever_tool(pdf_retriever, "economic_data_search", "Use for questions about India's economy, policies, and analyses from official reports.")
    sql_agent_executor = create_sql_agent(llm, db=db, agent_type="openai-tools", verbose=False)
    sql_tool = Tool(name="database_search", func=sql_agent_executor.invoke, description="Use for questions about specific numbers of business establishments or G-Sec auctions.")
    
    # --- NEW FORECASTING TOOL ---
    forecasting_tool = Tool(
        name="gsdp_forecaster",
        func=get_gdp_forecast,
        description="Use this tool when the user asks for a forecast or prediction of future GSDP for a specific Indian state. The input should be the name of the state."
    )
    
    tools = [pdf_search_tool, sql_tool, forecasting_tool]

    # --- CREATE THE AGENT ---
    agent_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", "You are an expert financial assistant. You have access to tools for answering questions and making forecasts. Use the tools to find the information and then answer the question. For forecasts, mention that they are based on historical data and not financial advice."),
            ("human", "{input}"),
            ("placeholder", "{agent_scratchpad}"),
        ]
    )
    agent = create_tool_calling_agent(llm, tools, agent_prompt)
    agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

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
                response = agent_executor.invoke({"input": user_query})
                answer = response.get("output", "I encountered an error.")
                st.session_state.chat_history.append(AIMessage(content=answer))
                st.rerun() # Rerun to display the new message
else:
    st.info("Please enter your Groq API Key in the sidebar to start.")