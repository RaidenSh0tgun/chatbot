import pandas as pd
from tqdm import tqdm
from langchain_ollama.llms import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings

# ----------- SETTINGS -----------
input_file = 'test_list.csv'
output_file = 'spaa_test_results.csv'
model_name = "gemma3:4b"
embedding_model = "mxbai-embed-large"  # Must match the one used to build the DB
db_directory = "./chroma_db"

# ----------- INITIALIZE RAG COMPONENTS -----------
print("Connecting to Vector Database...")
embeddings = OllamaEmbeddings(model=embedding_model)
vector_db = Chroma(persist_directory=db_directory, embedding_function=embeddings)

# Configure retriever to get top 3 relevant chunks
retriever = vector_db.as_retriever(search_kwargs={"k": 3})

print("Initializing Model...")
model = OllamaLLM(model=model_name)

template = """
Your name is Friday. You are a helpful assistant of School of Public Affairs and Administration (SPAA) at Rutgers University-Newark. 

Your duty is to answer questions related to the School based ONLY on the provided information.

Always answer in 1-5 sentences. Be clear, direct, and to the point.

Related information: 
{info}

Question: {question}
"""

prompt = ChatPromptTemplate.from_template(template)
chain = prompt | model

# ----------- LOAD TEST DATA -----------
try:
    df = pd.read_csv(input_file)
except FileNotFoundError:
    print(f"Error: Could not find {input_file}")
    exit()

# ----------- RUN TESTING -----------
for run in range(1, 4):
    column_name = f'answer_{run}'
    source_column = f'sources_{run}' # Optional: save sources for verification
    print(f"\nStarting Run {run}/3...")
    
    generated_answers = []
    retrieved_sources = []
    
    for index, row in tqdm(df.iterrows(), total=len(df), desc=f"Run {run}"):
        question = row['question']
        
        try:
            # 1. Retrieve documents
            docs = retriever.invoke(question)
            
            # 2. Extract text and source URLs
            context_text = "\n\n".join([doc.page_content for doc in docs])
            sources = ", ".join(list(set([doc.metadata.get('source', 'N/A') for doc in docs])))
            
            # 3. Generate response
            result = chain.invoke({
                "info": context_text, 
                "question": question
            })
            
            generated_answers.append(result.strip())
            retrieved_sources.append(sources)
            
        except Exception as e:
            print(f"Error on question '{question}': {e}")
            generated_answers.append("ERROR")
            retrieved_sources.append("ERROR")

    # Save results and sources for this run
    df[column_name] = generated_answers
    df[source_column] = retrieved_sources

# ----------- SAVE RESULTS -----------
# Use utf-8-sig so Excel opens the file with correct characters
df.to_csv(output_file, index=False, encoding='utf-8-sig')
print(f"\nTesting Complete!")
print(f"Results saved to: {output_file}")