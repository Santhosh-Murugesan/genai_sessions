from sentence_transformers import SentenceTransformer
import json
import faiss

# Download from the 🤗 Hub
model = SentenceTransformer("google/embeddinggemma-300m")

print("Model downloaded successfully. \n")

def create_query_embeddings(query):
    query_embeddings = model.encode_query(query)

    query_embeddings = query_embeddings.reshape(1, -1)

    print("Query embeddings created successfully.")
    print("Query embedding shape:", query_embeddings.shape)

    return query_embeddings

def find_similar_documents(query_embeddings, index, k=5):
    distances, indices = index.search(query_embeddings, k=k)

    print("Similarity scores calculated successfully. \n")

    print(f"Distances: {distances}")
    print(f"Indices: {indices}")

    return distances, indices

def fetch_document_by_id(id):
    with open("document.json", "r") as f:
        document_dict = json.load(f)
    
    print(f"Document with ID {id} fetched successfully. \n")
    print("---------------------------------")
    print(document_dict.get(str(id)))
    print("---------------------------------")
    return document_dict.get(str(id), "Document not found")

def relevant_chunks(query):

    #step 3: create query embeddings
    query_embeddings = create_query_embeddings(query) #[-0.0123456789, 0.123456789, ...]

    #step 4: load database embeddings from faiss index
    index = faiss.read_index("C:\\Users\\Santhosh\\OneDrive\\Documents\\GenAI_Workspace\\genai_sessions\\embeddings_with_vector_stores\\document_embeddings.faiss")

    #print(index)
    
    #step 5: get the similarity scores of all the chunks in the database.
    distances, indices = find_similar_documents(query_embeddings, index, k=5) #tensor([[0.327940, 0.294255, 0.607475, 0.494079,........0.272311]])

    #step 7: fetch the chunks corresponding to the top n similarity scores and its ids
    relevent_chunks = []
    for id in indices[0]:
        chunk = fetch_document_by_id(id)
        relevent_chunks.append(chunk)
    return relevent_chunks
