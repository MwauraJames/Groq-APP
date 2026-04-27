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
    api_messages = [{"role": "system", "content": "You are a helpful analyst. Answer using the provided context or use your tools if needed."}]
    api_messages.extend(st.session_state.conversation_history[:-1]) # Add past history (excluding the current prompt we just added)
    api_messages.append({"role": "user", "content": augmented_prompt}) # Add the newly augmented prompt

    with st.spinner("Agent 1 is thinking..."):
        # ==========================================
        # 5. AGENT 1: THE SEARCHER (Tool Calling Logic)
        # ==========================================
        response = client.chat.completions.create(
            model="openai/gpt-oss-120b",
            messages=api_messages,
            tools=tools,
            tool_choice="auto",
        )
        
        response_message = response.choices[0].message
        
        # Check if Agent 1 wants to use Wikipedia
        if response_message.tool_calls:
            api_messages.append(response_message) # Save the AI's request to use the tool
            
            for tool_call in response_message.tool_calls:
                if tool_call.function.name == "search_wikipedia":
                    func_args = json.loads(tool_call.function.arguments)
                    wiki_data = search_wikipedia(func_args.get("query"))
                    
                    # Feed the Wikipedia data back into the message list
                    api_messages.append({
                        "tool_call_id": tool_call.id,
                        "role": "tool",
                        "name": "search_wikipedia",
                        "content": wiki_data,
                    })
                    
            # Let Agent 1 generate a draft answer now that it has the Wikipedia data
            draft_response = client.chat.completions.create(
                model="openai/gpt-oss-120b",
                messages=api_messages,
            )
            draft_answer = draft_response.choices[0].message.content
        else:
            # If no tools were needed, this is the draft answer
            draft_answer = response_message.content

    with st.spinner("Agent 2 is evaluating..."):
        # ==========================================
        # 6. AGENT 2: THE EVALUATOR (Corrective Logic)
        # ==========================================
        evaluator_prompt = f"""
        You are a strict grading system. 
        User Question: {user_text}
        Draft Answer: {draft_answer}
        
        Does the Draft Answer directly address the User Question without hallucinating harmful or wildly inaccurate information? 
        Reply with exactly one word: PASS or FAIL.
        """
        
        eval_response = client.chat.completions.create(
            model="openai/gpt-oss-120b",
            messages=[{"role": "user", "content": evaluator_prompt}],
            temperature=0.0 # Force the model to be strict and consistent
        )
        
        grade = eval_response.choices[0].message.content.strip().upper()
        
        # Determine the final output based on Agent 2's grade
        if "PASS" in grade:
            final_reply = draft_answer
        else:
            print(f"[SYSTEM] Draft failed evaluation. Draft was: {draft_answer}")
            final_reply = "I apologize, but my internal evaluation systems flagged my drafted response for potential inaccuracies. Could you please rephrase your question or be more specific?"

    # Print the final result to the screen
    with st.chat_message('assistant'):
        st.markdown(final_reply)
        
    st.session_state.conversation_history.append({"role": "assistant", "content": final_reply})