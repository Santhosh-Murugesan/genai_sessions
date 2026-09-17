import sqlite3
import pandas as pd
from google import genai
from dotenv import load_dotenv
import os

load_dotenv()

#connect database
connection = sqlite3.connect("expenses.db")


client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

def call_gemini_api(prompt):
    response = client.models.generate_content(
    model="models/gemini-3.5-flash", 
    contents=prompt    
)
    
    return response.text

def covert_text_query_into_sql_query(user_input):

    prompt = f"""You are an SQL Professional your job is to convert text query into sql query to execute in my SQLite database
    ### DATABASE Details
    Columns : ID,Date,Category,Amount,Description,Created At
    Category : Transport,Shopping,Others,Food,Travel,Entertainment,Housing,Education
    
    ### Note 
    1.Generate only SQL Query According to Text query
    2.Do not add any symbols like !@#$%^&*()
    3.Ensure strictly return only the Equivalent SQL query without any extra contents

    ### INPUT FORMAT
    What is the total spend of transport ?
    
    ### OUTPUT FORMAT
    SELECT amount
    FROM expenses
    WHERE category = 'Transport'

    ### TEXT QUERY
    Convert following text query into SQL Query : {user_input}
    """
    sql_query = call_gemini_api(prompt)

    print(sql_query)

    return sql_query

def query_to_sqlite_database(query,connection):

    result = pd.read_sql_query(query, connection)

    return result.to_dict()


while True:
    #step 1: asking the user for a query
    user_input = input("Enter your query :  ")

    #step 2: we need sql query for my text query from the database
    query_repsonse = covert_text_query_into_sql_query(user_input)

    #step 3 : we need to get records from database
    database_result = query_to_sqlite_database(query_repsonse,connection)

    #step 3: augment the user input with the relevant chunks
    prompt = f"context: {database_result}\n\n Answer the following question based on the above provided context only: {user_input}"
    if user_input.lower() == 'exit':
        break

    print("Augumented prompt created successfully. \n")
    print('=========================================================================')
    print(prompt)
    print('=========================================================================')
    
    #step 9: call the Gemini API with the prompt to get final response
    response = call_gemini_api(prompt)
    print("Response from Gemini API:", response)

connection.close()
print("\nDatabase connection closed.")

