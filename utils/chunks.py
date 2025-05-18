from typing import List
from langchain.text_splitter import RecursiveCharacterTextSplitter

def chunk_manual(text: str, chunk_size: int = 500, overlap: int = 100) -> List[str]:
    """
    將文字切成多段 chunk, 用於 embedding
    
    Args:
        text (str): 完整文字
        chunk_size (int): 每段 chunk 長度
        overlap (int): 每段 chunk 之間要重疊多少字 (保持語意連貫)
    
    Returns:
        List[str]: 切割後的多個 chunk 段落
    """
    chunks = []
    start = 0
    text_length = len(text)

    while start < text_length:
        end = min(start+chunk_size, text_length)
        chunk = text[start:end].strip()

        if chunk:
            chunks.append(chunk)
    
        start += (chunk_size - overlap)

    return chunks

def chunk_with_langchain():
    pass