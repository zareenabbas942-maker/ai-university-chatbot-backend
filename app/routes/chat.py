from fastapi import APIRouter
from pydantic import BaseModel
from dotenv import load_dotenv
import os

from langchain_google_genai import ChatGoogleGenerativeAI

from app.rag.loader import load_university_data
from app.rag.splitter import split_university_data
from app.rag.vector_store import UniversityVectorStore


load_dotenv()

router = APIRouter()


class ChatRequest(BaseModel):
    message: str


# ============================================================
# LOAD UNIVERSITY DATA
# ============================================================

data = load_university_data()

chunks = split_university_data(data)

vector_store = UniversityVectorStore(chunks)


# ============================================================
# GEMINI CONFIGURATION
# ============================================================

api_key = os.getenv("GOOGLE_API_KEY")

llm = None

if api_key:
    llm = ChatGoogleGenerativeAI(
        model="gemini-3.8-flash",
        google_api_key=api_key,
        temperature=0.2,
        timeout=30,
        max_retries=0,
    )


# ============================================================
# LCWU / UNIVERSITY RELATED KEYWORDS
# ============================================================

UNIVERSITY_KEYWORDS = [
    "lcwu",
    "lahore college for women",
    "university",
    "admission",
    "admissions",
    "apply",
    "application",
    "program",
    "programs",
    "degree",
    "degrees",
    "faculty",
    "faculties",
    "department",
    "departments",
    "course",
    "courses",
    "scholarship",
    "scholarships",
    "financial aid",
    "fee",
    "fees",
    "tuition",
    "hostel",
    "hostels",
    "quota",
    "merit",
    "eligibility",
    "eligible",
    "requirement",
    "requirements",
    "campus",
    "location",
    "address",
    "contact",
    "website",
    "office",
    "registrar",
    "student",
    "students",
    "academic",
    "academics",
    "semester",
    "undergraduate",
    "postgraduate",
    "bs",
    "ms",
    "mphil",
    "phd",
]


# ============================================================
# CHECK WHETHER QUESTION IS UNIVERSITY RELATED
# ============================================================

def is_university_question(question: str) -> bool:

    normalized_question = question.lower().strip()

    if "lcwu" in normalized_question:
        return True

    if "lahore college for women" in normalized_question:
        return True

    for keyword in UNIVERSITY_KEYWORDS:

        if keyword in normalized_question:
            return True

    return False


# ============================================================
# CHECK WHETHER RETRIEVED CONTEXT MATCHES THE QUESTION
# ============================================================

def context_matches_question(
    question: str,
    context: str
) -> bool:

    question_lower = question.lower()
    context_lower = context.lower()

    # ========================================================
    # FEE / TUITION QUESTIONS
    # ========================================================

    fee_keywords = [
        "fee",
        "fees",
        "tuition",
        "cost",
        "charges",
        "expense",
    ]

    if any(keyword in question_lower for keyword in fee_keywords):

        return any(
            keyword in context_lower
            for keyword in fee_keywords
        )

    # ========================================================
    # SCHOLARSHIP QUESTIONS
    # ========================================================

    scholarship_keywords = [
        "scholarship",
        "scholarships",
        "financial aid",
    ]

    if any(
        keyword in question_lower
        for keyword in scholarship_keywords
    ):

        return (
            "scholarships and financial aid" in context_lower
            or "scholarship" in context_lower
            or "financial aid" in context_lower
        )

    # ========================================================
    # HOSTEL QUESTIONS
    # ========================================================

    hostel_keywords = [
        "hostel",
        "hostels",
        "accommodation",
    ]

    if any(
        keyword in question_lower
        for keyword in hostel_keywords
    ):

        return "hostel" in context_lower

    # ========================================================
    # QUOTA QUESTIONS
    # ========================================================

    quota_keywords = [
        "quota",
        "reserved seats",
        "reserved quota",
    ]

    if any(
        keyword in question_lower
        for keyword in quota_keywords
    ):

        return "quota" in context_lower

    # ========================================================
    # ADMISSION PROCESS QUESTIONS
    # ========================================================

    admission_process_keywords = [
        "admission process",
        "how to apply",
        "how do i apply",
        "application process",
        "apply for admission",
        "admission procedure",
    ]

    if any(
        keyword in question_lower
        for keyword in admission_process_keywords
    ):

        return (
            "admission process" in context_lower
            or "admission procedure" in context_lower
        )

    # ========================================================
    # PROGRAM QUESTIONS
    # ========================================================

    program_keywords = [
        "program",
        "programs",
        "degree",
        "degrees",
        "what does lcwu offer",
        "what programs",
    ]

    if any(
        keyword in question_lower
        for keyword in program_keywords
    ):

        return (
            "undergraduate programs" in context_lower
            or "programs" in context_lower
        )

    # ========================================================
    # FACULTY / DEPARTMENT QUESTIONS
    # ========================================================

    faculty_keywords = [
        "faculty",
        "faculties",
        "department",
        "departments",
    ]

    if any(
        keyword in question_lower
        for keyword in faculty_keywords
    ):

        return (
            "faculties and departments" in context_lower
            or "faculty" in context_lower
            or "department" in context_lower
        )

    # ========================================================
    # LOCATION / CONTACT QUESTIONS
    # ========================================================

    location_keywords = [
        "location",
        "address",
        "contact",
        "phone",
        "telephone",
        "website",
        "where is lcwu",
    ]

    if any(
        keyword in question_lower
        for keyword in location_keywords
    ):

        return (
            "location" in context_lower
            or "address" in context_lower
            or "contact" in context_lower
            or "website" in context_lower
        )

    # ========================================================
    # GENERAL LCWU QUESTION
    # ========================================================

    return True


# ============================================================
# SAFE NO-INFORMATION RESPONSE
# ============================================================

def no_information_response():

    return {
        "reply": (
            "I don't have this information in my current "
            "LCWU knowledge base. Please check the official "
            "LCWU website for the latest information."
        ),
        "sources": []
    }


# ============================================================
# CHAT API
# ============================================================

@router.post("/chat")
def chat(request: ChatRequest):

    question = request.message.strip()

    # ========================================================
    # EMPTY QUESTION
    # ========================================================

    if not question:

        return {
            "reply": "Please enter a question about LCWU.",
            "sources": []
        }

    # ========================================================
    # OUT-OF-SCOPE CHECK
    # ========================================================

    if not is_university_question(question):

        return {
            "reply": (
                "I'm an AI University Assistant for "
                "Lahore College for Women University (LCWU). "
                "I can help with LCWU admissions, programs, "
                "faculties, departments, scholarships, hostel, "
                "quota, contact information, and other "
                "university-related information."
            ),
            "sources": []
        }

    # ========================================================
    # RETRIEVE RELEVANT UNIVERSITY INFORMATION
    # ========================================================

    relevant_chunks = vector_store.search(
        question,
        top_k=1
    )

    # ========================================================
    # NO INFORMATION FOUND
    # ========================================================

    if not relevant_chunks:

        return no_information_response()

    # ========================================================
    # CREATE CONTEXT
    # ========================================================

    context = "\n\n".join(relevant_chunks)

    # ========================================================
    # CHECK WHETHER RETRIEVED DATA MATCHES QUESTION
    # ========================================================

    if not context_matches_question(
        question,
        context
    ):

        return no_information_response()

    # ========================================================
    # CREATE SOURCE INFORMATION
    # ========================================================

    source_list = [
        vector_store.get_source(chunk)
        for chunk in relevant_chunks
    ]

    # ========================================================
    # GEMINI NOT AVAILABLE
    # ========================================================

    if llm is None:

        return {
            "reply": context,
            "sources": source_list
        }

    # ========================================================
    # AI PROMPT
    # ========================================================

    prompt = f"""
You are an AI University Assistant for
Lahore College for Women University (LCWU).

Answer the student's question using ONLY the LCWU
knowledge base provided below.

Rules:
- Only answer questions related to LCWU.
- Use only information present in the provided knowledge base.
- Do not invent information.
- Do not make assumptions.
- Do not provide approximate information.
- Keep the answer clear and concise.
- Use bullet points when listing multiple items.
- If the exact information requested by the student is not present
  in the knowledge base, say exactly:

"I don't have this information in my current LCWU knowledge base."

- Never use information from outside the provided context.

Student question:
{question}

LCWU knowledge base:
{context}
"""

    # ========================================================
    # GEMINI REQUEST
    # ========================================================

    try:

        response = llm.invoke(prompt)

        # ====================================================
        # CLEAN GEMINI RESPONSE
        # ====================================================

        answer = response.content

        if isinstance(answer, dict):

            answer = answer.get("text", "")

        elif isinstance(answer, list):

            texts = []

            for item in answer:

                if isinstance(item, dict):

                    text_value = item.get("text", "")

                    if text_value:
                        texts.append(str(text_value))

                elif isinstance(item, str):

                    texts.append(item)

            answer = " ".join(texts)

        answer = str(answer).strip()

        # ====================================================
        # SUCCESSFUL RESPONSE
        # ====================================================

        if answer:

            return {
                "reply": answer,
                "sources": source_list
            }

    except Exception:

        pass

    # ========================================================
    # SAFE FALLBACK
    # ========================================================

    return {
        "reply": context,
        "sources": source_list
    }