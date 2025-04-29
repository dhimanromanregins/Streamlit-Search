import streamlit as st
from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance
from sentence_transformers import SentenceTransformer
from functools import lru_cache

# ---- Cached Model Loader ----
@lru_cache()
def get_embedding_model():
    return SentenceTransformer('all-MiniLM-L6-v2')

# ---- Qdrant Client Initialization ----
try:
    qdrant = QdrantClient(
    url=st.secrets["QDRANT_URL"],
    api_key=st.secrets["QDRANT_API_KEY"]
)
except Exception as e:
    st.error("Failed to connect to Qdrant.")
    raise e

# ---- Setup Collection ----
def setup_collection(collection_name="docs"):
    try:
        # Check if the collection already exists
        collections = qdrant.get_collections()
        if collection_name not in collections:
            print(f"Collection '{collection_name}' not found. Creating collection...")
            qdrant.recreate_collection(
                collection_name=collection_name,
                vectors_config=VectorParams(size=384, distance=Distance.COSINE),
            )
            print(f"Collection '{collection_name}' created successfully.")
        else:
            print(f"Collection '{collection_name}' already exists.")
    except Exception as e:
        st.error(f"Error setting up Qdrant collection: {e}")
        raise e

# ---- Insert Documents into Qdrant ----
def insert_documents(docs, collection_name="docs"):
    if not docs:
        print("No documents to insert.")
        return

    try:
        model = get_embedding_model()
        vectors = model.encode(docs).tolist()
        payload = [{"text": doc} for doc in docs]

        qdrant.upsert(
            collection_name=collection_name,
            points=[
                {"id": i, "vector": vectors[i], "payload": payload[i]}
                for i in range(len(docs))
            ]
        )
        print(f"Inserted {len(docs)} documents into '{collection_name}'.")
    except Exception as e:
        st.error(f"Error inserting documents: {e}")
        raise e

# ---- Search Documents in Qdrant ----
def search(query, collection_name="docs", top_k=5):
    try:
        model = get_embedding_model()
        vector = model.encode(query).tolist()
        search_result = qdrant.search(
            collection_name=collection_name,
            query_vector=vector,
            limit=top_k
        )
        return [hit.payload['text'] for hit in search_result]
    except Exception as e:
        st.error(f"Error during search: {e}")
        return []
