from sentence_transformers import SentenceTransformer

def get_embedder_by_sentence_transformer(model_name: str = 'all-MiniLM-L6-v2', device: str = 'cpu'):
    return SentenceTransformer(model_name).to(device)


if __name__ == '__main__':
    pass