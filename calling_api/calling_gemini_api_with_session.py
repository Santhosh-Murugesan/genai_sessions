from google import genai
from dotenv import load_dotenv
import os
import json

load_dotenv()

client = genai.Client(
    api_key=os.getenv("GEMINI_API_KEY")
)

def call_gemini_api(chat_history):
    response = client.models.generate_content(
        model="gemini-3.5-flash",
        contents=chat_history
    )

    return response.text

chat_history = []

while True:

    user_input = input(
        "\nEnter your prompt (or type 'exit' to quit): "
    )

    if user_input.lower() == "exit":
        break

    chat_history.append({
        "role": "user",
        "parts": [
            {
                "text": user_input
            }
        ]
    })
    with open("chat_history.json", "w") as file:
        json.dump(chat_history, file, indent=4)

    response = call_gemini_api(chat_history)

    chat_history.append({
        "role": "model",
        "parts": [
            {
                "text": response
            }
        ]
    })

    with open("chat_history.json", "w") as file:
        json.dump(chat_history, file, indent=4)

    print("\nGemini:", response)
