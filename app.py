import os
import gradio as gr
from dotenv import load_dotenv

from youtube_transcript_api import YouTubeTranscriptApi, NoTranscriptFound, TranscriptsDisabled
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq

load_dotenv()

# Global dictionary to cache vector stores
vector_stores = {}

# Initialize embeddings model
embeddings = HuggingFaceEmbeddings(
    model_name='BAAI/bge-base-en-v1.5',
    model_kwargs={'device': 'cpu'}
)


def process_video(video_id):
    """Process a YouTube video: fetch transcript, chunk, and embed."""
    video_id = video_id.strip()
    if not video_id:
        return "❌ Please enter a video ID."

    if video_id in vector_stores:
        return "✅ Video already processed! You can start asking questions."

    try:
        # Fetch transcript
        try:
            yt = YouTubeTranscriptApi()
            transcript_list = yt.list(video_id)
        except Exception as e:
            if "blocking requests" in str(e) and os.path.exists('cookies.txt'):
                import requests, http.cookiejar
                session = requests.Session()
                cookie_jar = http.cookiejar.MozillaCookieJar('cookies.txt')
                cookie_jar.load(ignore_discard=True, ignore_expires=True)
                session.cookies.update(cookie_jar)
                yt = YouTubeTranscriptApi(http_client=session)
                transcript_list = yt.list(video_id)
            else:
                raise e

        try:
            _transcript_obj = transcript_list.find_manually_created_transcript(
                [t.language_code for t in transcript_list]
            )
        except NoTranscriptFound:
            _transcript_obj = next(iter(transcript_list))

        _original_lang = _transcript_obj.language_code
        if _original_lang != 'en':
            _transcript_obj = _transcript_obj.translate('en')

        fetched = _transcript_obj.fetch()
        transcript = ' '.join(chunk['text'] if isinstance(chunk, dict) else chunk.text for chunk in fetched)

        if not transcript.strip():
            return "❌ Transcript is empty."

        # Split text
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        docs = text_splitter.create_documents([transcript])

        # Create vector store
        vector_store = FAISS.from_documents(docs, embeddings)
        vector_stores[video_id] = vector_store

        return "✅ Video processed successfully! You can now ask questions."

    except TranscriptsDisabled:
        return "❌ Transcripts are disabled for this video."
    except NoTranscriptFound:
        return "❌ No transcript found for this video."
    except Exception as e:
        return f"❌ Error: {str(e)}"


def ask_question(video_id, question, chat_history):
    """Ask a question about the processed video."""
    video_id = video_id.strip()
    question = question.strip()

    if not video_id or not question:
        chat_history.append((question, "Please enter both a video ID and a question."))
        return chat_history, ""

    if video_id not in vector_stores:
        chat_history.append((question, "⚠️ Please process the video first before asking questions."))
        return chat_history, ""

    try:
        retriever = vector_stores[video_id].as_retriever(search_type='similarity', search_kwargs={'k': 5})
        retrieved_docs = retriever.invoke(question)
        context_text = '\n\n'.join(doc.page_content for doc in retrieved_docs)

        prompt_template = """You are a helpful assistant.
Answer ONLY from the provided transcript context.
If the context is insufficient, just say "Nothing related to your question is discussed in this video."

{context}
Question: {question}"""

        prompt = prompt_template.format(context=context_text, question=question)

        # Initialize LLM
        llm = ChatGroq(
            model='mixtral-8x7b-32768',
            temperature=0.3
        )
        try:
            llm = ChatGroq(model='openai/gpt-oss-120b', temperature=0.3)
        except Exception:
            pass

        response = llm.invoke(prompt)
        chat_history.append((question, response.content))

    except Exception as e:
        chat_history.append((question, f"❌ Error: {str(e)}"))

    return chat_history, ""


# --- Gradio UI ---

theme = gr.themes.Soft(
    primary_hue="purple",
    secondary_hue="pink",
)

with gr.Blocks(theme=theme, title="YouTube RAG Chatbot", css="""
    .header { text-align: center; margin-bottom: 10px; }
    .header h1 { margin: 0; }
    footer { display: none !important; }
""") as demo:

    gr.HTML("""
        <div class="header">
            <h1>🎬 YouTube RAG Chatbot</h1>
            <p>Ask questions about any YouTube video — answers grounded in the transcript.</p>
        </div>
    """)

    with gr.Row():
        with gr.Column(scale=1):
            video_id_input = gr.Textbox(
                label="YouTube Video ID",
                placeholder="e.g. dQw4w9WgXcQ",
                info="The part after 'v=' in the YouTube URL"
            )
            process_btn = gr.Button("🔄 Process Video", variant="primary")
            status_output = gr.Textbox(label="Status", interactive=False, lines=2)
            thumbnail = gr.HTML(value="<div style='text-align:center; color:#999; padding:40px;'>Thumbnail will appear here</div>")

        with gr.Column(scale=2):
            chatbot = gr.Chatbot(label="Chat", height=400, type="tuples")
            question_input = gr.Textbox(
                label="Your Question",
                placeholder="Ask something about the video...",
                lines=1
            )
            ask_btn = gr.Button("💬 Ask", variant="primary")

    # --- Event handlers ---

    def update_thumbnail(video_id):
        video_id = video_id.strip()
        if video_id:
            return f'<div style="text-align:center;"><img src="https://img.youtube.com/vi/{video_id}/hqdefault.jpg" style="border-radius:12px; max-width:100%; box-shadow: 0 4px 15px rgba(0,0,0,0.15);" /></div>'
        return "<div style='text-align:center; color:#999; padding:40px;'>Thumbnail will appear here</div>"

    process_btn.click(
        fn=process_video,
        inputs=[video_id_input],
        outputs=[status_output]
    ).then(
        fn=update_thumbnail,
        inputs=[video_id_input],
        outputs=[thumbnail]
    )

    ask_btn.click(
        fn=ask_question,
        inputs=[video_id_input, question_input, chatbot],
        outputs=[chatbot, question_input]
    )

    question_input.submit(
        fn=ask_question,
        inputs=[video_id_input, question_input, chatbot],
        outputs=[chatbot, question_input]
    )

if __name__ == '__main__':
    demo.launch(server_name="0.0.0.0", server_port=int(os.environ.get('PORT', 7860)))
