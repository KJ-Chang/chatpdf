import pymupdf

def extract_text_from_pdf(pdf_path: str) -> str:
    """
    從 PDF 檔案中擷取存文字

    Args:
        pdf_path (str): PDF 檔案路徑

    Returns:
        str: PDF 內容所有頁面文字合併後的純文字
    """
    doc = pymupdf.open(pdf_path)
    full_text = []
    
    for page in doc:
        text = page.get_text()
        full_text.append(text)

    doc.close()
    return '\n'.join(full_text)
        


if __name__ == '__main__':
    pdf_path = '../data/eng_edu.pdf'
    texts = extract_text_from_pdf(pdf_path)
    print(texts)
    
