from google import genai
from dotenv import load_dotenv
import os

load_dotenv()


client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

with open("Prithviraj_Sukumaran_10_Popular_Movies.txt", "r") as file:
    external_context = file.read()

def call_gemini_api(prompt):
    
    response = client.models.generate_content(
    model="models/gemini-3.5-flash", 
    contents=prompt    
)
    
    return response.text

while True:
    user_input = input("Enter your prompt (or type 'exit' to quit): ")

    prompt = f"context: {external_context}\n\n Answer the following question based on the above provided context only: {user_input}"
    #prompt = user_input

    #print("Prompt sent to Gemini API:", prompt)

    if user_input.lower() == 'exit':
        break
    response = call_gemini_api(prompt)
    print("Response from Gemini API:", response)


