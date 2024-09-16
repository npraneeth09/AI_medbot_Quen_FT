from src.helper import load_pdf, text_split, download_hugging_face_embedding
from langchain_community.vectorstores import Qdrant

embeddings = download_hugging_face_embedding()

documents = load_pdf('Data')
texts = text_split(documents)


url = "http://localhost:6333" 
qdrant = Qdrant.from_documents(
    texts, 
    embeddings, 
    url=url,
    prefer_grpc=False,
    collection_name="vector_db"
)

print("Vector DB Successfully Created!")