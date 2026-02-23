import os
from typing import List, cast
from dotenv import load_dotenv
import google.generativeai as genai

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.documents import Document

load_dotenv()


class RAG:
    """Retrieval Augmented Generation using Gemini."""

    def __init__(self, vector_store):
        self.vector_store = vector_store

        genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

        self.model = genai.GenerativeModel("models/gemini-2.5-flash")

    def generate_answer(self, query: str, k: int = 4) -> str:

        docs = self.vector_store.similarity_search(query, k=k)

        if not docs:
            return "No relevant information found. Please upload a document first."

        context = "\n\n".join(
            [
                f"Document: {doc.metadata.get('source', 'Unknown')}\n{doc.page_content}"
                for doc in docs
            ]
        )

        prompt = f"""
You are a helpful assistant.

Answer ONLY using the provided context.
If answer is not in context, say: "Answer not found in document."

Context:
{context}

Question:
{query}
"""

        try:
            response = self.model.generate_content(prompt)
            return cast(str, response.text)

        except Exception as e:
            return f"Model Error: {str(e)}"