from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

# Load
loader = TextLoader("notes.txt")
documents = loader.load()

# Split
splitter = RecursiveCharacterTextSplitter(
    chunk_size=300,
    chunk_overlap=50,
)

chunks = splitter.split_documents(documents)

# Embeddings
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# Vector Store
vector_store = FAISS.from_documents(
    chunks,
    embeddings
)

# Retriever
retriever = vector_store.as_retriever(
    search_kwargs={"k": 3}
)

query = input("Ask: ")

docs = retriever.invoke(query)

for i, doc in enumerate(docs, start=1):
    print(f"\n========== Chunk {i} ==========")
    print(doc.page_content)