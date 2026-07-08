from dotenv import load_dotenv
from langchain_mistralai import MistralAIEmbeddings

load_dotenv()
emb = MistralAIEmbeddings(model='mistral-embed')
v = emb.embed_query('concert de jazz a Lille')
print('Dimension de l embedding :', len(v))
