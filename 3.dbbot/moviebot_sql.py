import pandas as pd
import sqlite3
from langchain.llms import OpenAI
from langchain.utilities import SQLDatabase
from langchain_experimental.sql import SQLDatabaseChain

# File path of the CSV file
file_path = 'C:/Users/alakh/Desktop/Generative AI Workshop/generativeai/3.dbbot/3idiots.csv'

#creating database

# SQLite database file path
database_path = 'database.db'

def convert_csv_db(csv_path, db_path):

    # Create a connection to the SQLite database
    connection = sqlite3.connect(db_path)


    # Read the CSV files into pandas DataFrames
    movie_df = pd.read_csv(csv_path)

    # Execute SQL CREATE TABLE statement to create the movie table

    create_table_sql = '''
    CREATE TABLE IF NOT EXISTS Movie (Actors text, Directed text , Produced text , Cinematographer text , Release_Date text);
    '''

    # Execute the CREATE TABLE statement
    connection.execute(create_table_sql)

    # Insert the data from DataFrames into the table
    movie_df.to_sql('3idiots', connection, if_exists='append', index=False)

    # Commit the changes and close the connection
    connection.commit()
    connection.close()

    return

convert_csv_db(file_path, database_path)

# this will load 'YOUR_OPENAI_API_KEY' from the .env file
from dotenv import load_dotenv
import os

load_dotenv()
openai_api_key = os.environ.get('OPENAI_API_KEY')

movie_db = SQLDatabase.from_uri('sqlite:///database.db')
english_sql_llm = OpenAI(temperature=0) # here the temperature is set to zero as we do not need any variability

#create a langchain with one node(english_sql_llm)
movie_sql_chain = SQLDatabaseChain.from_llm(llm = english_sql_llm,
                                     db = movie_db,
                                     verbose=True)

movie_sql_chain.run("Who are the actors in this movie")
movie_sql_chain.run("When was the movie released?")