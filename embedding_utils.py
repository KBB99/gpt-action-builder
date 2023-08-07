import openai
from openai.embeddings_utils import get_embedding, cosine_similarity

def generate_embedding(input):
    return openai.Embedding.create(input=input, model="text-embedding-ada-002")['data'][0]['embedding']
    
