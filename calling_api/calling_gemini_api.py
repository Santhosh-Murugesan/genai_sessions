from google import genai
from dotenv import load_dotenv
import os

load_dotenv()


client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

def call_gemini_api(prompt):
    response = client.models.generate_content(
    model="models/gemini-3.5-flash", 
    contents=prompt    
)
    
    return response.text

while True:
    user_input = input("Enter your prompt (or type 'exit' to quit): ")
    if user_input.lower() == 'exit':
        break
    response = call_gemini_api(user_input)
    print("Response from Gemini API:", response)


