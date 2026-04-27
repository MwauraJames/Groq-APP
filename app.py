import streamlit as st
import os
import json
from groq import Groq
import wikipedia
import chromadb
import PyPDF2
from sentence_transformers import SentenceTransformer
from langchain_text_splitters import RecursiveCharacterTextSplitter

# ==========================================
# 1. INITIALIZATION & CACHING
# ==========================================
@st.cache_resource
def init_system():
    ai_client = Groq()
    embed_model = SentenceTransformer('all-MiniLM-L6-v2')
    db_client = chromadb.PersistentClient(path='./my_local_data')
    db_collection = db_client.get_or_create_collection(name='document_workspace')
    return ai_client, embed_model, db_collection

client, embedding_model, collection = init_system()

# ==========================================
# 2. THE WIKIPEDIA TOOL DEFINITION
# ==========================================
def search_wikipedia(query: str):
    print(f"\n[SYSTEM] Executing tool: Searching Wikipedia for '{query}'...")
    try:
        search_results = wikipedia.search(query)
        if not search_results:
            return json.dumps({"error":"No Wikipedia pages found."})
        page_summary = wikipedia.summary(search_results[0], sentences=3)
        return json.dumps({"title": search_results[0], 'summary': page_summary})
    except wikipedia.exceptions.DisambiguationError as e:
        return json.dumps({"error": f"Query too broad. Specify one of these: {e.options[:5]}"})
    except Exception as e:
        return json.dumps({"error": str(e)})

tools = [{
    "type": "function",
    "function": {
        "name": "search_wikipedia",
        "description": "Searches Wikipedia for a topic and returns a brief summary. Use this to find factual information, history, or definitions.",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "The specific topic to search for."}
            },
            "required": ["query"],
        },
    },
}]

# ==========================================
# 3. UI & DOCUMENT UPLOAD SIDEBAR
# ==========================================
st.title("Enterprise RAG Assistant")

with st.sidebar:
    st.header("Document Knowledge Base")
    uploaded_file = st.file_uploader("Upload a PDF to the database", type=["pdf"])
    
    if uploaded_file is not None:
        if st.button("Process & Store Document"):
            with st.spinner("Extracting text..."):
                pdf_reader = PyPDF2.PdfReader(uploaded_file)
                raw_text = "".join(page.extract_text() + "\n" for page in pdf_reader.pages if page.extract_text())
                
            with st.spinner("Chunking with LangChain..."):
                text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150, separators=["\n\n", "\n", " ", ""])
                chunks = text_splitter.split_text(raw_text)
            
            with st.spinner("Embedding and Saving to ChromaDB..."):
                embeddings = embedding_model.encode(chunks).tolist()
                ids = [f"{uploaded_file.name}_chunk_{i}" for i in range(len(chunks))]
                collection.add(embeddings=embeddings, documents=chunks, ids=ids)
            st.success(f"Successfully processed {len(chunks)} chunks into the database!")
    # ... your existing upload and chunking code ...
    
    st.divider() # Adds a clean visual separating line in the sidebar
    st.subheader("Database Management")
    
    # The Danger Button
    if st.button("Clear Database (Wipe All Data)"):
        with st.spinner("Wiping ChromaDB..."):
            # 1. Fetch every single chunk currently in the database
            all_existing_data = collection.get()
            doc_ids = all_existing_data.get("ids", [])
            
            # 2. If the database isn't empty, delete all of those IDs
            if doc_ids:
                collection.delete(ids=doc_ids)
                st.success(f"Successfully deleted {len(doc_ids)} chunks. The database is completely empty.")
            else:
                st.info("The database is already empty.")
    if st.button("Clear Chat History"):
        st.session_state.conversation_history = []
        st.success("Chat history wiped. You can start a new conversation.")
# ==========================================
# 4. CHAT INTERFACE & MEMORY
# ==========================================
if "conversation_history" not in st.session_state:
    st.session_state.conversation_history = []

for message in st.session_state.conversation_history:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if user_text := st.chat_input("Ask a question about your documents or search Wikipedia..."):

    with st.chat_message("user"):
        st.markdown(user_text)
    st.session_state.conversation_history.append({"role": "user", "content": user_text})

    # --- VECTOR DATABASE SEARCH ---
    question_embedding = embedding_model.encode([user_text]).tolist()
    if collection.count() > 0:
        results = collection.query(query_embeddings=question_embedding, n_results=3)
        retrieved_context = "\n\n".join(results['documents'][0])
    else:
        retrieved_context = "No documents have been uploaded to the database yet."
    
    augmented_prompt = f"Context from local database:\n{retrieved_context}\n\nUser Question: {user_text}"
    
    # Compile the API payload
    api_messages = [{"role": "system", "content": "You are a helpful analyst. Answer using the provided context or use your tools if needed.Be creative but factual.Use your general knowledge where necessary."}]
    api_messages.extend(st.session_state.conversation_history[:-1]) # Add past history (excluding the current prompt we just added)
    api_messages.append({"role": "user", "content": augmented_prompt}) # Add the newly augmented prompt

        # ==========================================
    # 5 & 6. THE MULTI-AGENT FEEDBACK LOOP
    # ==========================================
    MAX_RETRIES = 3
    attempt = 0
    passed_evaluation = False
    final_reply = ""

    while attempt <= MAX_RETRIES and not passed_evaluation:
        with st.spinner(f"Agent 1 is drafting (Attempt {attempt + 1})..."):
            
            # --- AGENT 1 DRAFTS ---
            response = client.chat.completions.create(
                model="openai/gpt-oss-120b", 
                messages=api_messages,
                tools=tools,
                tool_choice="auto",
            )
            
            response_message = response.choices[0].message
            
            # Handle tool calling (Wikipedia)
            if response_message.tool_calls:
                api_messages.append(response_message)
                for tool_call in response_message.tool_calls:
                    if tool_call.function.name == "search_wikipedia":
                        func_args = json.loads(tool_call.function.arguments)
                        wiki_data = search_wikipedia(func_args.get("query"))
                        api_messages.append({
                            "tool_call_id": tool_call.id,
                            "role": "tool",
                            "name": "search_wikipedia",
                            "content": wiki_data,
                        })
                
                # Regenerate draft with the new Wikipedia data
                draft_response = client.chat.completions.create(
                    model="openai/gpt-oss-120b",
                    messages=api_messages,
                )
                draft_answer = draft_response.choices[0].message.content
            else:
                draft_answer = response_message.content

        with st.spinner("Agent 2 is evaluating..."):
            
            # Extract a readable string of the last 4 messages for the evaluator
            recent_history = st.session_state.conversation_history[-5:-1]
            history_text = "\n".join([f"{msg['role'].capitalize()}: {msg['content']}" for msg in recent_history])
            
            # --- AGENT 2 EVALUATES ---
            evaluator_prompt = f"""
            You are a strict grading system. 
            
            Recent Chat History:
            {history_text}
            
            User's Current Question: {user_text}
            Draft Answer: {draft_answer}
            
            Does the Draft Answer directly address the User Question accurately based on the chat history and provided context? 
            If it is good, reply ONLY with 'PASS'.
            If it is bad, reply with 'FAIL: [Explain exactly what is missing or wrong]'.
            """
            
            eval_response = client.chat.completions.create(
                model="openai/gpt-oss-120b",
                messages=[{"role": "user", "content": evaluator_prompt}],
                temperature=0.0 
            )
            
            grade = eval_response.choices[0].message.content.strip()
            
            # --- THE ROUTING DECISION ---
            if grade.startswith("PASS"):
                final_reply = draft_answer
                passed_evaluation = True
            else:
                attempt += 1
                print(f"\n[SYSTEM] Attempt {attempt} Failed. Reason: {grade}")
                
                if attempt <= MAX_RETRIES:
                    # Give Agent 1 the draft and the feedback, and force a rewrite
                    correction_prompt = f"Your previous answer was rejected. The evaluator said: {grade}\n\nPlease rewrite your answer to fix these specific issues."
                    api_messages.append({"role": "assistant", "content": draft_answer})
                    api_messages.append({"role": "user", "content": correction_prompt})
                else:
                    final_reply = "I apologize, but my internal safety systems could not verify a highly accurate answer after multiple attempts. Please try rephrasing your question."

    # Print the final result to the screen
    with st.chat_message('assistant'):
        st.markdown(final_reply)
        
    st.session_state.conversation_history.append({"role": "assistant", "content": final_reply})