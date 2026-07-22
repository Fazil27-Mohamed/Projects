from pypdf import PdfReader
from chunker import split_text
from embedding import create_vector_db
import os


def load_all_pdfs(data_folder):

    text = ""

    pdf_files = [f for f in os.listdir(data_folder) if f.endswith(".pdf")]

    print(f"Found {len(pdf_files)} PDF files")

    for pdf in pdf_files:

        pdf_path = os.path.join(data_folder, pdf)

        print(f"Reading: {pdf}")

        reader = PdfReader(pdf_path)

        for page in reader.pages:

            page_text = page.extract_text()

            if page_text:
                text += page_text + "\n"

    return text


if __name__ == "__main__":

    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_dir = os.path.dirname(current_dir)

    data_folder = os.path.join(project_dir, "data")

    content = load_all_pdfs(data_folder)

    chunks = split_text(content)

    print(f"\nTotal Chunks: {len(chunks)}")

    create_vector_db(chunks)

    print("\n✅ All PDFs Indexed Successfully!")