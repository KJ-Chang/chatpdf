import torch
from utils.chunks import chunk_manual
from src.pdf_reader import extract_text_from_pdf
from src.embedder import get_embeddings_by_sentence_transformer


def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'Using device: {device}')

    pdf_path = 'data/eng_edu.pdf'

    text = extract_text_from_pdf(pdf_path)
    chunks = chunk_manual(text, 500, 50)
    

    embeddings = get_embeddings_by_sentence_transformer(chunks, 'sentence-transformers/distiluse-base-multilingual-cased-v1')
    

if __name__ == '__main__':
    main()