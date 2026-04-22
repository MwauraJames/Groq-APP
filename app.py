import streamlit as st
import os
from groq import Groq
import chromadb
import PyPDF2
from sentence_transformers import SentenceTransformer
from langchain_text_splitters import RecursiveCharacterTextSplitter

@st.cache_resource
def init_system():
    ai_client=Groq()

    embed_model=SentenceTransformer('all-MiniLM-L6-v2')

    db_client=chromadb.PersistentClient(path='./my_local_data')
    db_collection=db_client.get_or_create_collection(name='document_workspace')

    return ai_client, embed_model,db_collection

client, embedding_model, collection=init_system()

st.title("Enterprise RAG Assistant")

with st.sidebar:
    st.header("Document Knowledge Base")
    uploaded_file = st.file_uploader("Upload a PDF to the database", type=["pdf"])
    
    if uploaded_file is not None:
        if st.button("Process & Store Document"):
            with st.spinner("Extracting text..."):
                # Read the PDF directly from memory
                pdf_reader = PyPDF2.PdfReader(uploaded_file)
                raw_text = ""
                for page in pdf_reader.pages:
                    if page.extract_text():
                        raw_text += page.extract_text() + "\n"
                
            with st.spinner("Chunking with LangChain..."):
                # Context-Aware Chunking: Splits by paragraphs, then sentences
                text_splitter = RecursiveCharacterTextSplitter(
                    chunk_size=1000,
                    chunk_overlap=150,
                    separators=["\n\n", "\n", " ", ""]
                )
                chunks = text_splitter.split_text(raw_text)
            
            with st.spinner("Embedding and Saving to ChromaDB..."):
                # Convert chunks to vectors and save
                embeddings = embedding_model.encode(chunks).tolist()
                
                # Generate unique IDs for each chunk
                ids = [f"{uploaded_file.name}_chunk_{i}" for i in range(len(chunks))]
                
                collection.add(
                    embeddings=embeddings,
                    documents=chunks,
                    ids=ids
                )
            st.success(f"Successfully processed {len(chunks)} chunks into the database!")

if "conversation_history" not in st.session_state:
    st.session_state.conversation_history = []

for message in st.session_state.conversation_history:
    if message["role"] != "system":
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

if user_text := st.chat_input("Ask a question about your documents..."):

    with st.chat_message("user"):
        st.markdown(user_text)
        
    st.session_state.conversation_history.append({"role": "user", "content": user_text})

    question_embedding=embedding_model.encode([user_text]).tolist()

    if collection.count() > 0:
        results = collection.query(
            query_embeddings=question_embedding,
            n_results=3 # Retrieve top 3 most relevant chunks
        )
        retrieved_context = "\n\n".join(results['documents'][0])
    else:
        retrieved_context = "No documents have been uploaded to the database yet."
    
    augmented_prompt = f"Context Information:\n{retrieved_context}\n\nUser Question: {user_text}"
    system_instruction = "You are an expert analyst. Answer the user's question using ONLY the provided context information. If the context does not contain the answer, explicitly state that."

    # CORRECTED: Wrap the text in the required dictionary list format
    formatted_messages = [{"role": "system", "content": system_instruction}]
    
    # Add all previous chat history (excluding the very first system prompt if you stored one)
    for msg in st.session_state.conversation_history:
        if msg["role"] != "system":
            formatted_messages.append(msg)
            
    # Finally, add the current augmented RAG prompt
    formatted_messages.append({"role": "user", "content": augmented_prompt})

    response = client.chat.completions.create(
        messages=formatted_messages, # Pass the list, not the raw string
        #max_tokens=1000,
        model="llama-3.1-8b-instant", 
    )

    # CORRECTED: Changed 'chat_completion' to 'response' to match the variable above
    ai_reply = response.choices[0].message.content

    with st.chat_message('assistant'):
        st.markdown(ai_reply)
    st.session_state.conversation_history.append({"role": "assistant", "content": ai_reply})
