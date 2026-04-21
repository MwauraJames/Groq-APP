import streamlit as st
import os
from groq import Groq

# 1. Set up the web page title
st.title("Groq AI Assistant")

# 2. Initialize the Groq client
# (Streamlit will look for GROQ_API_KEY in the environment)
client = Groq()

# 3. Initialize memory using session_state so it doesn't get erased
if "conversation_history" not in st.session_state:
    st.session_state.conversation_history = [
        {"role": "system", "content": "You are a helpful coding assistant. Keep answers brief."}
    ]

# 4. Draw all past messages to the screen (skipping the hidden system prompt)
for message in st.session_state.conversation_history:
    if message["role"] != "system":
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

# 5. Create the chat input box at the bottom of the screen
if user_text := st.chat_input("Type your message here..."):
    
    # Immediately display the user's message on screen
    with st.chat_message("user"):
        st.markdown(user_text)
        
    # Append the user's message to the memory
    st.session_state.conversation_history.append({"role": "user", "content": user_text})

    # Call the Groq API
    chat_completion = client.chat.completions.create(
        messages=st.session_state.conversation_history,
        max_tokens=1000,
        model="llama-3.1-8b-instant", 
    )

    # Extract the AI's reply
    ai_reply = chat_completion.choices[0].message.content

    # Display the AI's reply on screen
    with st.chat_message("assistant"):
        st.markdown(ai_reply)

    # Append the AI's reply to the memory
    st.session_state.conversation_history.append({"role": "assistant", "content": ai_reply})