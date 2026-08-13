from google import genai
from dotenv import load_dotenv
import os
from genai_sessions.creating_embeddings.create_embeddings_with_gemma import relevant_chunks

load_dotenv()


client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

def call_gemini_api(prompt):
    response = client.models.generate_content(
    model="models/gemini-3.5-flash", 
    contents=prompt    
)
    
    return response.text

while True:
    #step 1: asking the user for a query
    user_input = input("Enter your query :  ")

    #step 2: we need relevant chunks for my query from the document embeddings database
    relevent_chunks = relevant_chunks(user_input)

    #please look at create_embeddings_with_gemma.py file for steps 3 to 7. In this file we are only focusing on steps 8 and 9.

    #step 8: augment the user input with the relevant chunks
    prompt = f"context: {relevent_chunks}\n\n Answer the following question based on the above provided context only: {user_input}"
    if user_input.lower() == 'exit':
        break

    print("Augumented prompt created successfully. \n")
    print('=========================================================================')
    print(prompt)
    print('=========================================================================')
    
    #step 9: call the Gemini API with the prompt to get final response
    response = call_gemini_api(prompt)
    print("Response from Gemini API:", response)


