# Replace 'YOUR_API_KEY' with your actual OpenAI API key
import os
os.environ['OPENAI_API_KEY'] = 'YOUR_API_KEY' 

from langchain.llms import OpenAI #Importing the OpenAI LLM(Large Language Model)

movie_bot_llm = OpenAI(temperature = 0.6)
# creates an instance of the OpenAI language model. The temperature parameter is being set to 0.6,
# which is a hyperparameter that controls the randomness of the model's output. Higher values (e.g., 1.0)
# make the output more random, while lower values (e.g., 0.2) make it more deterministic

movie_name = movie_bot_llm("I want to make an Action movie. Suggest a catchy name for this.")
# OpenAI language model will use this prompt to generate a response
print(f"Movie name: {movie_name}")

movie_actors_str = movie_bot_llm(f"Suggest some actors for '{movie_name}'. Return it as a comma separated list.") 
movie_actors_list = [actor.strip() for actor in movie_actors_str.split(',')]
print(f"\nMovie actors: {movie_actors_list}")

movie_scene = movie_bot_llm(f"Write a small opening scene for this movie: '{movie_name}'.") 
print(f"\nMovie scene: {movie_scene}")

## Making a movie poster ##

# Replace 'YOUR_API_KEY' with your actual OpenAI API key
import openai
openai.api_key = 'YOUR_API_KEY'

def movie_poster(movie_scene):
  prompt = f"A digital art that depicts the scene: {movie_scene}"
  response = openai.Image.create(
    prompt=prompt,
    n=1 # n is the number of images
  )
  movie_poster_url = response["data"][0]["url"]
  return movie_poster_url

movie_poster_url = movie_poster(movie_scene)
print(f"\nMovie poster URL: {movie_poster_url}")