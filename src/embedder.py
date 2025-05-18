from typing import List
from sentence_transformers import SentenceTransformer

def get_embeddings_by_sentence_transformer(
        sentences: List[str],
        model_name: str = 'all-MiniLM-L6-v2', 
        device: str = 'cpu',
        ):
    
    st_model = SentenceTransformer(model_name).to(device)
    embeddings = st_model.encode(sentences, convert_to_numpy=True)

    return embeddings


if __name__ == '__main__':
    pass