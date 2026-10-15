# YouTube RAG Chatbot 🎬🤖

A Retrieval-Augmented Generation (RAG) chatbot that lets you ask questions about any YouTube video. It fetches the video's transcript, chunks and embeds it, then uses an LLM to answer questions grounded strictly in the transcript.

![Python](https://img.shields.io/badge/Python-3.9+-blue?logo=python&logoColor=white)
![Flask](https://img.shields.io/badge/Flask-Web_App-green?logo=flask&logoColor=white)
![LangChain](https://img.shields.io/badge/LangChain-RAG-orange)

## Features

- 🎯 **Transcript-grounded answers** — responses are based only on the video content, no hallucination
- 🌐 **Multi-language support** — auto-translates non-English transcripts to English
- ⚡ **In-memory FAISS vector store** — fast similarity search, no external database needed
- 🖥️ **Clean web UI** — split-pane chat interface with video thumbnail preview
- 📓 **Jupyter notebook included** — step-by-step walkthrough of the full RAG pipeline

## Tech Stack

| Component | Technology |
|-----------|-----------|
| Backend | Flask |
| Embeddings | HuggingFace (`BAAI/bge-base-en-v1.5`) |
| Vector Store | FAISS |
| LLM | Groq API (Mixtral 8x7B) |
| Transcript | `youtube-transcript-api` |
| Framework | LangChain |

## Quick Start

### 1. Clone the repo

```bash
git clone https://github.com/<your-username>/youtube-rag-chatbot.git
cd youtube-rag-chatbot
```

### 2. Create a virtual environment (recommended)

```bash
python -m venv venv
source venv/bin/activate   # Linux/Mac
venv\Scripts\activate      # Windows
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Set up your API key

```bash
cp .env.example .env
```

Edit `.env` and add your [Groq API key](https://console.groq.com/keys):

```
GROQ_API_KEY=gsk_your_actual_key_here
```

### 5. Run the app

```bash
python app.py
```

Open [http://127.0.0.1:5000](http://127.0.0.1:5000) in your browser.

### 6. Use it

1. Enter a YouTube video ID (the part after `v=` in the URL, e.g. `dQw4w9WgXcQ`)
2. Wait for the transcript to be processed and embedded
3. Ask questions about the video!

## Notebook

The [`youtube_rag_chatbot.ipynb`](youtube_rag_chatbot.ipynb) notebook walks through each stage of the RAG pipeline step by step:

1. **Transcript fetching** — download and translate the transcript
2. **Text splitting** — chunk the transcript with overlap
3. **Embedding & indexing** — embed chunks into a FAISS vector store
4. **Retrieval** — similarity search for relevant context
5. **Augmented generation** — prompt the LLM with retrieved context
6. **Full LCEL chain** — compose everything into a single LangChain chain

> **Note:** The notebook defaults to GPU (`device: 'cuda'`). Change to `'cpu'` if you don't have a CUDA GPU.

## Project Structure

```
├── app.py                      # Flask web application
├── templates/
│   └── index.html              # Frontend UI
├── youtube_rag_chatbot.ipynb   # Step-by-step notebook
├── requirements.txt            # Python dependencies
├── .env.example                # Template for environment variables
├── .gitignore                  # Git ignore rules
└── README.md                   # This file
```

## Cookie-based Transcript Fetching (Optional)

If YouTube blocks transcript requests (common on servers/VPNs), you can place a `cookies.txt` file (Netscape format) in the project root. The app will automatically use it as a fallback. You can export cookies from your browser using extensions like [Get cookies.txt LOCALLY](https://chromewebstore.google.com/detail/get-cookiestxt-locally/cclelndahbckbenkjhflpdbgdldlbecc).

> **⚠️ Never commit `cookies.txt` — it contains your YouTube session tokens. It is already in `.gitignore`.**

## License

MIT
