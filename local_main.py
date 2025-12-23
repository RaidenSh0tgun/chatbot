from langchain_ollama.llms import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.vectorstores import Chroma
from langchain_ollama import OllamaEmbeddings

# 1. Load the existing Vector Database
# IMPORTANT: Use the same embedding model you used in vector_db.py
embeddings = OllamaEmbeddings(model="mxbai-embed-large")
vector_db = Chroma(persist_directory="./chroma_db", embedding_function=embeddings)

# 2. Setup the Retriever
# search_kwargs={"k": 3} tells the bot to look at the top 3 most relevant chunks
retriever = vector_db.as_retriever(search_kwargs={"k": 3})

# 3. Setup the Model (Gemma 3)
model = OllamaLLM(model="gemma3:4b")

template = """
Your name is Friday. You are a helpful assistant of School of Public Affairs and Administration (SPAA) at Rutgers University-Newark. 

Your duty is to answer questions related to the School.
Always answer in 1-5 sentences unless the user explicitly requests more detail.
Be clear, direct, and only use the provided information.

Conversation History: {context}

Related Information: {info}

Question: {question}
"""

prompt = ChatPromptTemplate.from_template(template)
chain = prompt | model

# 4. Chat Loop
# Move 'context' OUTSIDE the loop so the bot actually remembers what you said previously
context = ""

print("Friday is online. Type 'q' to quit.")

while True:
    print("\n" + "-"*30)
    question = input("How may I help today? ")
    
    if question.lower() == "q":
        break
    
    # Retrieve relevant chunks from your 22k chunk database
    docs = retriever.invoke(question)
    info = "\n".join([doc.page_content for doc in docs])
    
    # Generate the result
    result = chain.invoke({"context": context, "info": info, "question": question})
    
    print(f"\nFriday: {result}")
    
    # Update context for the next turn
    context += f"\nUser: {question}\nFriday: {result}"