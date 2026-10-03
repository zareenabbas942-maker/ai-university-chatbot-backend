from fastapi import APIRouter
from pydantic import BaseModel
from dotenv import load_dotenv
from google import genai
import os

from app.rag.loader import load_university_data
from app.rag.splitter import split_university_data
from app.rag.vector_store import UniversityVectorStore


load_dotenv()

router = APIRouter()


class ChatRequest(BaseModel):
    message: str


# Load university data
data = load_university_data()

# Split data into sections
chunks = split_university_data(data)

# Create RAG vector store
vector_store = UniversityVectorStore(chunks)


# Load Gemini API key
api_key = os.getenv("GOOGLE_API_KEY")

client = None

if api_key:
    client = genai.Client(api_key=api_key)


@router.post("/chat")
def chat(request: ChatRequest):

    # Retrieve relevant university information
    relevant_chunks = vector_store.search(
        request.message,
        top_k=1
    )

    if not relevant_chunks:
        return {
            "reply": (
                "I don't have this information in my current "
                "LCWU knowledge base. Please check the official "
                "LCWU website for the latest information."
            ),
            "sources": []
        }

    # Combine retrieved information
    context = "\n\n".join(relevant_chunks)

    # If Gemini is not configured
    if client is None:
        return {
            "reply": (
                "The AI service is not configured correctly. "
                "Please check the GOOGLE_API_KEY."
            ),
            "sources": []
        }

    prompt = f"""
You are an AI University Assistant for Lahore College for Women University (LCWU).

Answer the student's question using ONLY the LCWU information provided below.

Rules:
- Do not invent information.
- Do not make assumptions.
- Give a natural, helpful answer.
- Keep the answer concise and easy for a university student to understand.
- Use bullet points when listing multiple items.
- If the information is not available in the provided knowledge base, say:
  "I don't have this information in my current LCWU knowledge base."
- Do not mention that you are using RAG or a knowledge base unless necessary.

Student question:
{request.message}

LCWU information:
{context}
"""

    try:
        print("Sending request to Gemini...")

        response = client.models.generate_content(
            model="gemini-3.8-flash",
            contents=prompt
        )

        answer = response.text

        print("Gemini response received successfully.")

        return {
            "reply": answer,
            "sources": []
        }

    except Exception as e:

        print("\n========== GEMINI ERROR ==========")
        print(repr(e))
        print("==================================\n")

        return {
            "reply": (
                "I couldn't generate the AI response right now. "
                "Please try again."
            ),
            "sources": []
        }