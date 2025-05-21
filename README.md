# ChatPDF

這是一個可以讓你跟 PDF 文件對話的專案！透過 AI 的加持，幫助你快速理解文件內容、回答各種問題。

![操作示範](data/MyChatPDF.gif)

## 這個專案可以做什麼？

- 上傳 PDF 文件讓系統讀取
- 用自然的方式問問題
- 系統會從文件中找出相關內容來回答你
- 回答很有智慧，不是只會複製貼上喔！

## 專案用了哪些技術？

- 前端介面：用 Gradio 來製作簡單好用的網頁
- 文件處理：使用 PyMuPDF 來讀取 PDF
- 文字搜尋：採用 FAISS-GPU 來找出相關段落
- AI 對話：使用 Falcon-7B-Instruct 大型語言模型
- 文字理解：透過 Sentence-Transformers 來處理

## 如何使用？

1. 執行系統：
```bash
python main.py
```

2. 使用步驟：
   - 第一步：上傳你的 PDF 檔案，等系統處理一下下
   - 第二步：輸入你想問的問題
   - 第三步：按下「提交問題」，就會得到回答囉！

## 專案架構說明

1. **PDF 處理模組** (src/pdf_reader.py)
   - 負責把 PDF 變成系統看得懂的文字

2. **向量處理模組** (src/embedder.py)
   - 把文字轉換成 AI 可以理解的格式

3. **搜尋模組** (src/vector_store.py)
   - 幫你找出最相關的文件內容

4. **回答模組** (src/generator.py)
   - 用 AI 產生人性化的回答

5. **輔助工具** (utils/)
   - chunks.py：幫忙切割文件內容
   - prompt.py：設計 AI 的回答方式