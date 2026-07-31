from langchain_huggingface import HuggingFaceEmbeddings

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

vector = embeddings.embed_query(
    "What is Binary Search?"
)

print(type(vector))
print(len(vector))
print(vector[:10])