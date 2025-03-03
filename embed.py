from langchain_community.document_loaders import JSONLoader
from sentence_transformers import SentenceTransformer
from langchain_chroma import Chroma
from langchain.docstore.document import Document
from typing import List
import json

class EmbeddingModel:
    def __init__(self, model):
        self.model = SentenceTransformer(model, trust_remote_code=True)
    
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return [self.model.encode(t).tolist() for t in texts]
            
    def embed_query(self, query: str) -> List[float]:
        return self.model.encode(query).tolist()
    
def custom_relevance_score_fn(similarity_score: float) -> float:
    # Example calculation (customize as needed)
    relevance_score = 1 / (1 + similarity_score)
    return relevance_score

def embed_docs(json_path:str):
    with open(json_path, 'r') as file:
        data = json.load(file)
    
    messages = []
    for doc in data['mess']:
        message_parts = []
        if 'time' in doc:
            message_parts.append("time: " + doc["time"] )
        if 'user' in doc:
            message_parts.append("user: " + doc['user'])
        if 'ai' in doc:
            message_parts.append("ai: " + doc['ai'] )
        
        # Join the parts with a newline and add to messages if any part exists
        if message_parts:
            messages.append(Document(page_content="\n".join(message_parts)))
    
    embeddings_fn = EmbeddingModel('all-MiniLM-L6-v2')
  
    vectorstore = Chroma.from_documents(
        documents=messages,
        embedding=embeddings_fn,
        persist_directory='conv_logs',
        relevance_score_fn=custom_relevance_score_fn
    )
    return vectorstore

if __name__=='__main__':
    vectorstore = embed_docs("text 1.json")    

