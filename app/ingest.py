import os
from pathlib import Path

from dotenv import load_dotenv
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
VECTOR_DIR = BASE_DIR / "vectorstore"

# Documents authorized for each role
ROLE_DOCUMENTS = {
    "admin": [
        "shipment_tracking.txt",
        "delivery_delay_sop.txt",
        "dispatch_escalation_policy.txt",
        "customer_support_policy.txt",
    ],
    "operations": [
        "shipment_tracking.txt",
        "delivery_delay_sop.txt",
        "dispatch_escalation_policy.txt",
    ],
    "dispatcher": [
        "shipment_tracking.txt",
        "dispatch_escalation_policy.txt",
    ],
    "customer_support": [
        "shipment_tracking.txt",
        "customer_support_policy.txt",
    ],
}

def main():

    api_key = os.getenv("GEMINI_API_KEY")

    if not api_key:
        raise RuntimeError("GEMINI_API_KEY missing from .env")

    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/gemini-embedding-2",
        google_api_key=api_key,
    )

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=700,
        chunk_overlap=100,
    )

    all_documents = {}

    # Load and split documents
    for file_name in set(
        file
        for files in ROLE_DOCUMENTS.values()
        for file in files
    ):

        file_path = DATA_DIR / file_name

        if not file_path.exists():
            print(f"Warning: Missing document: {file_name}")
            continue

        loader = TextLoader(
            str(file_path),
            encoding="utf-8",
        )

        documents = loader.load()

        chunks = splitter.split_documents(documents)

        all_documents[file_name] = chunks

        print(
            f"Loaded {file_name}: {len(chunks)} chunks"
        )

    # Create separate vectorstore for every role
    for role, allowed_files in ROLE_DOCUMENTS.items():

        role_documents = []

        for file_name in allowed_files:

            if file_name in all_documents:
                role_documents.extend(
                    all_documents[file_name]
                )

        if not role_documents:
            print(f"No documents available for {role}")
            continue

        role_vector_dir = VECTOR_DIR / role

        role_vector_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        vectorstore = FAISS.from_documents(
            role_documents,
            embeddings,
        )

        vectorstore.save_local(
            str(role_vector_dir)
        )

        print(
            f"\nSuccessfully created vectorstore for {role}"
        )

        print(
            f"Documents: {allowed_files}"
        )

        print(
            f"Chunks: {len(role_documents)}"
        )

        print(
            f"Saved to: {role_vector_dir}"
        )

    print("\nAll role-based vectorstores created successfully!")

if __name__ == "__main__":
    main()