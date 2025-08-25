# Economic Data Navigator 🇮🇳

An advanced, multi-tool AI agent built with Streamlit and LangChain to provide a conversational interface for analyzing and forecasting the Indian economy. This platform integrates multiple data sources and analytical models, allowing users to ask complex questions and receive intelligent, data-driven answers.

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://your-app-link.streamlit.app) ---

## 🚀 Core Features

The application is powered by a central LangChain agent that intelligently selects the best tool to answer a user's query. The agent has access to three specialized tools:

### 1. AI-Powered Document Q&A (RAG System)
* **Functionality**: Ask questions in plain English about official documents like RBI Annual Reports, Union Budget Speeches, and Economic Surveys.
* **How it Works**: This feature uses a Retrieval-Augmented Generation (RAG) pipeline. The `app.py` script creates a `pdf_search_tool` that connects to a Chroma vector store. This knowledge base is built by the `data_ingestion_and_cleaning.ipynb` notebook, which processes and embeds the raw PDF files.

### 2. Natural Language Database Querying (SQL Agent)
* **Functionality**: Get specific data points by asking questions about structured data, such as G-Sec auction results or state-wise economic indicators.
* **How it Works**: The `app.py` script initializes a `sql_tool` powered by a LangChain SQL Agent. This agent translates the user's question into a SQL query, executes it on the `esd_indicators.sqlite` database, and returns a natural language response. The database is populated by the `structured_data_sql.ipynb` and `add_auction_data_to_db.ipynb` notebooks.

### 3. Time-Series Forecasting (GSDP Prediction)
* **Functionality**: Request future Gross State Domestic Product (GSDP) forecasts for any Indian state (e.g., "What is the GSDP forecast for Maharashtra?").
* **How it Works**: The agent uses a dedicated `forecasting_tool` defined in `app.py`. This tool calls the `get_gdp_forecast` function, which trains a Prophet model on the fly using a master dataset created by `create_forecasting_dataset.ipynb` and `build_forecasting_model.ipynb`.

---

## 🛠️ Project Architecture & Workflow

The project is divided into a data processing backend (Jupyter Notebooks) and a user-facing frontend (Streamlit App). The LangChain agent acts as the brain connecting the two.

📂 Raw Data Files  
 ├── 📄 PDFs (RBI Reports, Budgets, etc.)  
 ├── 📊 CSVs (G-Sec Auctions, State Indicators)  
 │
 └── ⚙️ Data Processing Backend (/notebooks)  
      ├── 🧠 RAG Pipeline → `data_ingestion_and_cleaning.ipynb`  
      ├── 🗄️ SQL Pipeline → `structured_data_sql.ipynb`, `add_auction_data_to_db.ipynb`  
      ├── 📈 Forecasting Pipeline → `create_forecasting_dataset.ipynb`, `build_forecasting_model.ipynb`  
      │     └── 📦 Processed Assets (/data/processed, /vector_store)  
      │
      ├── 🧠 Chroma Vector Store  
      ├── 🗃️ SQLite Database → `esd_indicators.sqlite`  
      └── 📊 Forecasting Dataset → `final_forecasting_dataset.csv`  

➡️ Output → ✨ Frontend & Logic (`app.py`)  
   ├── 🎨 Streamlit UI (Chat Interface)  
   ├── 🧠 LangChain AgentExecutor (The Brain)  
   └── 🛠️ Tools (PDF Search, SQL Query, Forecasting)


---

## 💻 Tech Stack

-   **Frontend:** Streamlit
-   **AI & LLM Orchestration:** LangChain, Groq (Llama 3)
-   **Data Processing:** Pandas, NumPy, SQLAlchemy
-   **Time-Series Forecasting:** Prophet (from Meta)
-   **Vector DB & Embeddings:** ChromaDB, Hugging Face Transformers
-   **PDF Extraction:** PyMuPDF (`fitz`)

---

## ⚙️ Setup and Local Installation

To run this project on your local machine, follow these steps:

1.  **Clone the Repository:**
    ```sh
    git clone [https://github.com/aaryan-sky/economic-data-navigator.git](https://github.com/aaryan-sky/economic-data-navigator.git)
    cd economic-data-navigator
    ```

2.  **Create and Activate a Virtual Environment:**
    ```sh
    python -m venv .venv
    # On Mac/Linux: source .venv/bin/activate
    # On Windows: .\.venv\Scripts\activate
    ```

3.  **Install Dependencies:**
    ```sh
    pip install -r requirements.txt
    ```

4.  **Set Up API Key:**
    -   Get a free API key from [Groq](https://console.groq.com/keys).
    -   The application will ask for this key in the sidebar when you run it.

5.  **Run the Streamlit App:**
    ```sh
    streamlit run app.py
    ```
    Open your browser to the local URL provided by Streamlit.

---

## 📄 License

This project is licensed under the MIT License. See the `LICENSE` file for more details.

