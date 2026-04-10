from pypdf import PdfReader

def extract_text_from_pdf(pdf_path):
    reader = PdfReader(pdf_path)
    text = ""
    for page in reader.pages:
        text += page.extract_text() or ""
    return text

# Example usage:
if __name__ == "__main__":
    pdf_file = "msa1.pdf"
    text = extract_text_from_pdf(pdf_file)
    print(text)  # Print first 1000 characters