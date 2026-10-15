import os
from flask import Flask, request, jsonify, render_template
from dotenv import load_dotenv

from youtube_transcript_api import YouTubeTranscriptApi, NoTranscriptFound, TranscriptsDisabled
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq

load_dotenv()

app = Flask(__name__)
app.config['TEMPLATES_AUTO_RELOAD'] = True

# Global dictionary to cache vector stores
vector_stores = {}

# Initialize embeddings model (lightweight model for free-tier deployment)
embeddings = HuggingFaceEmbeddings(
    model_name='all-MiniLM-L6-v2',
    model_kwargs={'device': 'cpu'}
)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/process-video', methods=['POST'])
def process_video():
    data = request.json
    video_id = data.get('video_id')
    
    if not video_id:
        return jsonify({'success': False, 'error': 'No video_id provided'}), 400
        
    if video_id in vector_stores:
        return jsonify({'success': True, 'message': 'Video already processed'})
        
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
            return jsonify({'success': False, 'error': 'Transcript is empty'}), 400
            
        # Split text
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        docs = text_splitter.create_documents([transcript])
        
        # Create vector store
        vector_store = FAISS.from_documents(docs, embeddings)
        
        # Cache
        vector_stores[video_id] = vector_store
        
        return jsonify({'success': True, 'message': 'Video processed successfully'})
        
    except TranscriptsDisabled:
        return jsonify({'success': False, 'error': 'Transcripts are disabled for this video'}), 400
    except NoTranscriptFound:
        return jsonify({'success': False, 'error': 'No transcript found for this video'}), 400
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/ask', methods=['POST'])
def ask():
    data = request.json
    video_id = data.get('video_id')
    question = data.get('question')
    
    if not video_id or not question:
        return jsonify({'success': False, 'error': 'Missing video_id or question'}), 400
        
    if video_id not in vector_stores:
        return jsonify({'success': False, 'error': 'Video not processed yet'}), 400
        
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
        
        # Initialize LLM (in endpoint to avoid threading issues with Groq if any)
        llm = ChatGroq(
            model='mixtral-8x7b-32768', # using mixtral as fallback
            temperature=0.3
        )
        try:
            llm = ChatGroq(model='openai/gpt-oss-120b', temperature=0.3)
        except Exception:
            pass
            
        response = llm.invoke(prompt)
        
        return jsonify({
            'success': True,
            'answer': response.content
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(debug=False, host='0.0.0.0', port=port)
