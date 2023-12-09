# this will load 'YOUR_OPENAI_API_KEY' from the .env file
from dotenv import load_dotenv
import os

load_dotenv()
import openai
openai.api_key = os.environ.get('OPENAI_API_KEY')

from PyPDF2 import PdfReader

# mention the path for the pdf file - "Demo-pdf-3-idiots.pdf"
pdf_file_path = 'Demo-pdf-3-idiots.pdf' 

def read_movie_pdf_text(pdf_file_path):
    
    movie_pdf_reader = PdfReader(pdf_file_path) 

    # read data from the file line by line
    movie_pdf_text = ''
    for i, page in enumerate(movie_pdf_reader.pages):
        text = page.extract_text()
        if text:
            movie_pdf_text += text
    
    return movie_pdf_text

movie_pdf_text = read_movie_pdf_text(pdf_file_path)

# write the prompt which you want to ask from the pdf. For example - Here we are using: 'Brief the story of 3-idiots from'
prompt = f"Brief the story of 3-idiots from:\n\n{movie_pdf_text}"

# Call the OpenAI API to generate a response
response = openai.Completion.create(
        engine="text-davinci-003",
        prompt=prompt,
        max_tokens=256, # sets the maximum number of tokens for the response, You can adjust this value based on your desired output length
        temperature=0.1 # you can adjust this value within the range of 0 to 1; controls the diversity of the generated response.
        # Lower temperature results in more predictable response, while higher temperature results in more variable response.
    )

# Extract and print the generated result
result = response.choices[0].text.strip()
print(f"Answer: {result}")