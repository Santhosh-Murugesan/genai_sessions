from sentence_transformers import SentenceTransformer
import json

# Download from the 🤗 Hub
model = SentenceTransformer("google/embeddinggemma-300m")

print("Model downloaded successfully. \n")

def create_query_embeddings(query):
    query_embeddings = model.encode_query(query)

    print("Query embeddings created successfully. \n")

    return query_embeddings

def find_similar_documents(query_embeddings, document_embeddings):
    similarities = model.similarity(query_embeddings, document_embeddings)

    print("Similarity scores calculated successfully. \n")

    return similarities

def get_top_n_similarities(similarities, k=5):

    similarities_list = similarities[0]*100 # tensor([32.7940, 29.4255, 60.7475, 49.4079,........27.2311])

    similarities_list.tolist() # [32.79398727416992, ......354957580566406, 21.401302337646484, 27.231061935424805]

    #create a dictionary with id as key and similarity score as value
    similarity_score_dict = {}
    id = 1
    for chunk in similarities_list:
        similarity_score_dict[id] = chunk.tolist() # {"1": 32.79398727416992, "2": 29.425500869750977}
        id += 1

    #sort dictionary based on values in descending order 
    sorted_similarity_score_dict = {}
    for key in sorted(similarity_score_dict, key=similarity_score_dict.get, reverse=True):
        sorted_similarity_score_dict[key] = similarity_score_dict[key]

    #put 1st k highest similarity scores in a new dictionary
    top_n_similarity_score_dict = {}
    for key, value in sorted_similarity_score_dict.items():
        if len(top_n_similarity_score_dict) < k:
            top_n_similarity_score_dict[key] = value
        else:
            break 

    print(f"Top {k} similarity scores fetched successfully.")
    
    return top_n_similarity_score_dict

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

    #step 4: load database embeddings from json file
    with open("document_embeddings.json", "r") as f:
        document_embeddings = json.load(f)

    #step 5: get the similarity scores of all the chunks in the database.
    similarities = find_similar_documents(query_embeddings, document_embeddings)

    #step 6: get the top n similarity scores and their corresponding document ids
    top_n_similarity_score_dict = get_top_n_similarities(similarities, k=5)

    #step 7: fetch the chunks corresponding to the top n similarity scores and its ids
    relevent_chunks = []
    for id, score in top_n_similarity_score_dict.items():
        document = fetch_document_by_id(id)
        relevent_chunks.append(document) #[""This is the first chunk of the document.", "This is the second chunk of the document.", ...]

    return relevent_chunks
