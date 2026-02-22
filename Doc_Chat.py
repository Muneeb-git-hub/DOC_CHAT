import os
import streamlit as st
from langchain_huggingface import HuggingFaceEmbeddings, HuggingFaceEndpoint, ChatHuggingFace
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_classic.chains import RetrievalQA
from dotenv import load_dotenv

load_dotenv()

# ── Only functions here are ONLY because st.cache_resource requires it ──

@st.cache_resource
def get_vectorstore():
    embedding_model = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')
    db = FAISS.load_local("vectorstore/db_faiss", embedding_model, allow_dangerous_deserialization=True)
    return db

@st.cache_resource
def get_llm():
    llm_base = HuggingFaceEndpoint(
        repo_id="meta-llama/Meta-Llama-3.1-8B-Instruct",
        temperature=0.2,
        max_new_tokens=1024,
        task="text-generation"
    )
    return ChatHuggingFace(llm=llm_base)

# ───────────────────────────────────────────────────────────────

st.title("Ask Chatbot!")

if 'messages' not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    st.chat_message(message['role']).markdown(message['content'])

prompt = st.chat_input("Pass your prompt here")

if prompt:
    st.chat_message('user').markdown(prompt)
    st.session_state.messages.append({'role': 'user', 'content': prompt})

    CUSTOM_PROMPT_TEMPLATE = """
            Use the pieces of information provided in the context to answer user's question.
            If you dont know the answer, just say that you dont know, dont try to make up an answer. 
            Dont provide anything out of the given context
            Context: {context}
            Question: {question}
            Start the answer directly. No small talk please.
            """

    try:
        db = get_vectorstore()
        llm = get_llm()

        custom_prompt = PromptTemplate(
            template=CUSTOM_PROMPT_TEMPLATE,
            input_variables=["context", "question"]
        )

        qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=db.as_retriever(search_kwargs={'k': 3}),
            return_source_documents=False,
            chain_type_kwargs={'prompt': custom_prompt}
        )

        response = qa_chain.invoke({'query': prompt})
        result = response["result"]

        st.chat_message('assistant').markdown(result)
        st.session_state.messages.append({'role': 'assistant', 'content': result})

    except Exception as e:
        st.error(f"Error: {str(e)}")