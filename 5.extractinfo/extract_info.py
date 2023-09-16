# this will load 'YOUR_OPENAI_API_KEY' from the .env file
from dotenv import load_dotenv
import os

load_dotenv()
openai_api_key = os.environ.get('OPENAI_API_KEY')

from langchain.document_loaders import WikipediaLoader

# fetches information from Wikipedia about Will Smith. You can change this in the query parameter.
actor_docs = WikipediaLoader(query="Will Smith", load_max_docs=1).load()

from langchain.chat_models import ChatOpenAI
from langchain.chains import create_extraction_chain 

# define a schema(actor_schema), which is a structured way to specify the properties or pieces of information you want to extract from text data.
actor_schema = {
    "properties": {
        "name": {"type": "string"},
        "birth_date": {"type": "string"},
        "movie_names": {"type": "string"},
    },
}

# text from where we want to extract information
actor_text = actor_docs[0].page_content
print(f"About the actor: {actor_text}")

extract_info_llm = ChatOpenAI(temperature=0, model="gpt-3.5-turbo")
info_chain = create_extraction_chain(actor_schema, extract_info_llm)
actor_info = info_chain.run(actor_text)
print(f"\nActor info: {actor_info}")