from google import genai
from dotenv import load_dotenv
import streamlit as st
import os

load_dotenv()
st.title("Tejaswini AI Chat Bot")
st.subheader("Hi! I am your Tejaswini AI Chat Bot. Please enter your prompt below and I will respond accordingly.")

client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

def call_gemini_api(prompt):
    response = client.models.generate_content(
    model="models/gemini-flash-lite-latest", 
    contents=prompt    
)
    
    return response.text

user_input = st.text_input("Enter your query (or type 'exit' to quit): ")
response = call_gemini_api(user_input)
st.write("Response from Tejaswini AI:", response)


