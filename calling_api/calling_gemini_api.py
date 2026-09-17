from google import genai
from dotenv import load_dotenv
import os
from PIL import Image

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

    # 3. Load your image using PIL
    image = Image.open(r"C:\Users\Santhosh\OneDrive\Documents\GenAI_Workspace\genai_sessions\multimodel_emdedding_model\pdf_vector_store\images\page_2_image_1.jpeg")
    print(image)
    prompt = [image,user_input]
    print("Prompt to Gemini API:", prompt)
    response = call_gemini_api(prompt)  
    print("Response from Gemini API:", response)

