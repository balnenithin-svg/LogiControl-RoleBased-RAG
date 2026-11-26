
import os

from dotenv import load_dotenv
from supabase import create_client
from langchain_google_genai import GoogleGenerativeAIEmbeddings

load_dotenv()

# Server-side document permissions
ROLE_DOCUMENT_IDS = {
    "admin": [
        "LOG-TRACK-001",
        "LOG-DELAY-002",
        "LOG-DISPATCH-003",
        "LOG-CUSTOMER-004",
        "LOG-ADMIN-001",
    ],
    "operations": [
        "LOG-TRACK-001",
        "LOG-DELAY-002",
        "LOG-DISPATCH-003",
    ],
    "dispatcher": [
        "LOG-TRACK-001",
        "LOG-DISPATCH-003",
    ],
    "customer_support": [
        "LOG-TRACK-001",
        "LOG-CUSTOMER-004",
    ],
}


class SupabaseRAG:

    def __init__(self):
        supabase_url = os.getenv("SUPABASE_URL")
        supabase_key = os.getenv("SUPABASE_SECRET_KEY")
        gemini_key = os.getenv("GEMINI_API_KEY")

        if not supabase_url or not supabase_key:
            raise RuntimeError(
                "SUPABASE_URL or SUPABASE_SECRET_KEY is missing"
            )

        if not gemini_key:
            raise RuntimeError("GEMINI_API_KEY is missing")

        self.supabase = create_client(
            supabase_url,
            supabase_key,
        )

        self.embeddings = GoogleGenerativeAIEmbeddings(
            model="models/gemini-embedding-2",
            google_api_key=gemini_key,
            output_dimensionality=768,
        )

        print("Supabase RAG initialized successfully")

    def retrieve(self, question: str, role: str, limit: int = 5):

        role = role.lower().strip()

        if role not in ROLE_DOCUMENT_IDS:
            raise ValueError("Invalid user role")

        if not question or not question.strip():
            raise ValueError("Question cannot be empty")

        allowed_document_ids = ROLE_DOCUMENT_IDS[role]

        print(f"Supabase RAG: Searching for role {role}")

        # Generate query embedding
        query_embedding = self.embeddings.embed_query(
            question.strip()
        )

        if len(query_embedding) != 768:
            raise RuntimeError(
                "Query embedding must contain 768 dimensions"
            )

        # Call role-aware Supabase hybrid search
        response = self.supabase.rpc(
            "hybrid_search_knowledge_chunks_for_role",
            {
                "query_text": question.strip(),
                "query_embedding": query_embedding,
                "allowed_document_ids": allowed_document_ids,
                "match_count": limit,
            },
        ).execute()

        results = response.data or []

        # Additional application-side permission validation
        permitted_results = [
            item for item in results
            if item.get("document_id") in allowed_document_ids
        ]

        print(
            f"Supabase RAG: Retrieved {len(permitted_results)} "
            f"authorized chunks"
        )

        return permitted_results

