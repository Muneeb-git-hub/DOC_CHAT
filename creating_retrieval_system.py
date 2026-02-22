from langchain_huggingface import HuggingFaceEmbeddings,HuggingFaceEndpoint,ChatHuggingFace
from langchain_core.prompts import PromptTemplate
from langchain_community.vectorstores import FAISS
from langchain_classic.chains import RetrievalQA
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import  load_dotenv
load_dotenv()

llm_base = HuggingFaceEndpoint(
    repo_id="meta-llama/Meta-Llama-3.1-8B-Instruct",
    temperature=0.2,
    max_new_tokens=1024,
    task="text-generation"
)

llm = ChatHuggingFace(llm=llm_base)


# Step 2: Connect LLM with FAISS and Create chain

CUSTOM_PROMPT_TEMPLATE = """
Use the pieces of information provided in the context to answer user's question.
If you dont know the answer, just say that you dont know, dont try to make up an answer. 
Dont provide anything out of the given context

Context: {context}
Question: {question}

Start the answer directly. No small talk please.
"""

prompt=PromptTemplate(
    template=CUSTOM_PROMPT_TEMPLATE,
    input_variables=["context", "question"]
)

path="vectorstore/db_faiss"
embedding_model=HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
db=FAISS.load_local(path, embedding_model, allow_dangerous_deserialization=True)


qa_chain=RetrievalQA.from_chain_type(
    llm=llm,


    chain_type="stuff",
    retriever=db.as_retriever(search_kwargs={'k':3}),
    return_source_documents=True,
    chain_type_kwargs={'prompt':prompt}
)

user_query=input("Write Query Here: ")
response=qa_chain.invoke({'query': user_query})
print("RESULT: ", response["result"])
print("SOURCE DOCUMENTS: ", response["source_documents"])
# import google.generativeai as genai
# import os

# genai.configure(api_key='AIzaSyC3oxQav_bTNWfjf6TB4YKPyUcvH_Z_d_Q')

# for model in genai.list_models():
#     if 'generateContent' in model.supported_generation_methods:
#         print(model.name)