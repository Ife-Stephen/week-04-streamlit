# Week 4 – Session 8: Streamlit RAG Application

## Project Overview

This project builds a web-based Retrieval-Augmented Generation (RAG) application using Streamlit, Jina AI embeddings, and the Groq API.

Users can upload PDF or TXT documents, ask questions about their contents, and receive answers generated using retrieved document context.

## Features

- Upload PDF and TXT documents.
- Extract readable text from uploaded files.
- Split documents into chunks of approximately 800 characters with 100 characters of overlap.
- Generate embeddings using Jina AI.
- Retrieve the three most relevant chunks using cosine similarity.
- Generate context-grounded answers using Groq.
- Display retrieved source chunks and similarity scores.
- Maintain chat history using Streamlit session state.
- Store document chunks and embeddings in session state to avoid unnecessary re-embedding during normal app reruns.
- Handle file, API, and network errors.

## Technologies

- Python
- Streamlit
- Jina AI Embeddings API
- Groq API
- Requests
- Python-dotenv
- pypdf

## Project Structure

```text
week-04-streamlit/
│
├── app.py
├── rag.py
├── requirements.txt
├── README.md
├── .gitignore
└── .env
```

## Installation and Local Setup

### 1. Create a virtual environment

```powershell
python -m venv .venv
```

### 2. Activate the environment

```powershell
.\.venv\Scripts\Activate.ps1
```

### 3. Install dependencies

```powershell
python -m pip install -r requirements.txt
```

### 4. Configure API keys

Create a `.env` file in the project directory:

```text
GROQ_API_KEY=your_groq_api_key
JINA_API_KEY=your_jina_api_key
```

Never upload API keys or the `.env` file to GitHub.

### 5. Run the application

```powershell
python -m streamlit run app.py
```

## How It Works

1. The user uploads a PDF or TXT document.
2. The application extracts the document text.
3. The text is divided into overlapping chunks.
4. Jina AI generates embeddings for the chunks.
5. The application stores chunks and embeddings in Streamlit session state.
6. The user submits a question through the chat interface.
7. Jina generates a query embedding.
8. Cosine similarity identifies the most relevant document chunks.
9. Groq generates an answer using the retrieved context.
10. The application displays the answer and retrieved source chunks.

## Session State

Streamlit reruns the application when users interact with its interface.

The application uses `st.session_state` to preserve:

- Document chunks
- Document embeddings
- Uploaded document information
- Chat messages

A SHA-256 document hash is used to detect changes in uploaded files. This prevents the application from generating document embeddings again when the same document remains uploaded during normal interactions.

## PDF Support

The application uses `pypdf` to extract text from PDFs.

It supports text-based PDFs. Scanned PDFs containing only images require OCR, which is not included in this version.

## Security

API keys are loaded locally from `.env` and can also be supplied through Streamlit Community Cloud secrets.

The `.env`, virtual environment, and local Streamlit secrets files are excluded from Git tracking.

## Deployment

The application can be deployed using Streamlit Community Cloud by connecting its GitHub repository.

The Streamlit Cloud secrets configuration requires:

```toml
GROQ_API_KEY = "your_groq_api_key"
JINA_API_KEY = "your_jina_api_key"
```

The application's entrypoint file is `app.py`.

## Conclusion

Session 8 extends the Session 7 RAG pipeline into an interactive web application.

It allows users to upload PDF and TXT documents, ask questions, receive context-grounded answers, and inspect the retrieved source chunks.