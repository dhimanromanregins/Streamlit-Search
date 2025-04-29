import requests
import os
import streamlit as st

GROQ_API_KEY = st.secrets['GROQ_API_KEY']
GROQ_API_URL = st.secrets['GROQ_API_URL']

def ask_llama(context, query):
    prompt = f"Use the following context to answer the question:\n\n{context}\n\nQuestion: {query}"
    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json"
    }
    data = {
        "model": "llama3-70b-8192",
        "messages": [
            {"role": "user", "content": prompt}
        ]
    }
    response = requests.post(GROQ_API_URL, headers=headers, json=data)
    return response.json()['choices'][0]['message']['content']
