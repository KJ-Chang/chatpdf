import torch
import gradio as gr
from utils.chunks import chunk_manual
from utils.prompt import build_prompt
from src.pdf_reader import extract_text_from_pdf
from src.embedder import get_embedder_by_sentence_transformer
from src.generator import load_model_and_tokenizer, generate
from src.vector_store import create_faiss_index_native

class ChatPDFState:
    def __init__(self):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.chunks = None
        self.index = None
        self.embedder = None
        # 在初始化時就載入模型和tokenizer
        print("正在載入語言模型...")
        self.model, self.tokenizer = load_model_and_tokenizer('tiiuae/falcon-7b-instruct', self.device)
        print("語言模型載入完成！")

# 建立全域狀態
state = ChatPDFState()

def process_pdf(pdf_file):
    if pdf_file is None:
        return "請上傳PDF文件"
    
    # 讀取上傳的PDF文件
    text = extract_text_from_pdf(pdf_file.name)
    state.chunks = chunk_manual(text, 500, 50)
    
    # 創建文件的 embeddings
    state.embedder = get_embedder_by_sentence_transformer('sentence-transformers/all-mpnet-base-v2', state.device)
    document_embeddings = state.embedder.encode(state.chunks, batch_size=2, convert_to_numpy=True)
    
    # 創建向量索引
    state.index = create_faiss_index_native(document_embeddings, state.device)
    
    return "PDF文件已處理完成，現在可以開始提問"

def answer_question(question):
    if state.index is None or state.chunks is None or state.embedder is None:
        return "請先上傳並處理PDF文件"
    
    # 處理用戶問題
    query_embeddings = state.embedder.encode([question], batch_size=2, convert_to_numpy=True)
    
    # 找出相關文本
    contexts = []
    D, I = state.index.search(query_embeddings, k=5)
    for d, i in zip(D[0], I[0]):
        contexts.append(state.chunks[i])

    # 使用已載入的模型生成回答
    prompt = build_prompt(question, contexts)
    response = generate(state.model, state.tokenizer, prompt, state.device)
    
    return response

def create_interface():
    with gr.Blocks(title="MyChatPDF") as demo:
        gr.Markdown("# 📚 MyChatPDF")
        gr.Markdown("第一步：上傳 PDF 文件進行向量化處理")
        
        with gr.Row():
            pdf_input = gr.File(label="上傳 PDF 文件", file_types=[".pdf"])
            process_btn = gr.Button("處理文件")
        
        with gr.Row():
            process_output = gr.Textbox(label="處理狀態", lines=2)
            
        gr.Markdown("第二步：提出問題")
        
        with gr.Row():
            question_input = gr.Textbox(label="請輸入您的問題", placeholder="請問...")
            submit_btn = gr.Button("提交問題")
        
        with gr.Row():
            answer_output = gr.Textbox(label="回答", lines=5)
        
        process_btn.click(
            fn=process_pdf,
            inputs=[pdf_input],
            outputs=process_output
        )
        
        submit_btn.click(
            fn=answer_question,
            inputs=[question_input],
            outputs=answer_output
        )
    
    return demo

if __name__ == '__main__':
    demo = create_interface()
    demo.launch(share=False, server_name='0.0.0.0')  # 設置 share=False 來禁用 public URL