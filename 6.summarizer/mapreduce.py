#Install necessary libraries
#pip -q install langchain openai==0.27.0 tiktoken
#pip install cohere
#3pip install llmx
#Get API Key
from dotenv import load_dotenv
import os
load_dotenv()
openai_api_key = os.environ.get('OPENAI_API_KEY')

#Setting up Summarization Chain
from langchain import OpenAI, PromptTemplate, LLMChain
from langchain.text_splitter import CharacterTextSplitter
from langchain.chains.mapreduce import MapReduceChain
from langchain.prompts import PromptTemplate
llm = OpenAI(temperature=0)
#Load the Text splitter
text_splitter = CharacterTextSplitter()
#Load the doc
with open('paris.txt') as f:
  paris = f.read()
texts = text_splitter.split_text(paris)
#Length of texts
len(texts)
#Converting our text file to Docs
from langchain.docstore.document import Document
docs = [Document(page_content=t) for t in texts[:1]]
#Summarize with Map Reduce
from langchain.chains.summarize import load_summarize_chain
#To wrap the summary text
import textwrap
chain = load_summarize_chain(llm,chain_type="map_reduce")
output_summary = chain.run(docs)
wrapped_text = textwrap.fill(output_summary, width=100)
print(wrapped_text)
#For summarizing each part
chain.llm_chain.prompt.template
#For combining the parts
chain.combine_document_chain.llm_chain.prompt.template
#Executing Chains
chain = load_summarize_chain(llm,chain_type="map_reduce", verbose=True)
output_summary = chain.run(docs)
wrapped_text = textwrap.fill(output_summary,width=100,break_long_words=False ,replace_whitespace=False)
print(wrapped_text)