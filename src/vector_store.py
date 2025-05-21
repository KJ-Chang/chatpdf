import faiss

def create_faiss_index_native(documents, device):
    dimension = documents.shape[1]
    index = faiss.IndexFlatL2(dimension)

    if device == 'cuda':
        res = faiss.StandardGpuResources()
        gpu_index = faiss.index_cpu_to_gpu(res, 0, index)
        index = gpu_index
    
    index.add(documents)

    return index

def create_faiss_index_langchain():
    pass