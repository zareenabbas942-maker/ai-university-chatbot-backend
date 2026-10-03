import math
import re
from collections import Counter


class UniversityVectorStore:
    def __init__(self, chunks):
        self.chunks = chunks

    # ============================================================
    # SEARCH
    # ============================================================

    def search(self, query, top_k=1):

        query_lower = " ".join(
            query.lower().split()
        )

        # ========================================================
        # FEE / TUITION QUESTIONS
        # ========================================================

        fee_keywords = [
            "fee",
            "fees",
            "tuition",
            "tuition fee",
            "tuition fees",
            "cost",
            "cost of",
            "charges",
            "expense",
            "expenses",
            "semester fee",
            "admission fee",
            "fee structure",
            "fee structure for",
        ]

        if any(
            keyword in query_lower
            for keyword in fee_keywords
        ):

            fee_chunks = []

            for chunk in self.chunks:

                chunk_lower = chunk.lower()

                # Only return a section if it actually contains
                # fee information.
                if (
                    "fee:" in chunk_lower
                    or "fees:" in chunk_lower
                    or "tuition fee" in chunk_lower
                    or "tuition fees" in chunk_lower
                    or "fee structure" in chunk_lower
                ):

                    fee_chunks.append(chunk)

            # If no fee information exists in knowledge base,
            # return empty result instead of unrelated sections.
            if not fee_chunks:

                return []

            return fee_chunks[:top_k]

        # ========================================================
        # GENERAL UNIVERSITY INFORMATION
        # ========================================================

        general_info_keywords = [
            "contact",
            "contact number",
            "phone",
            "phone number",
            "telephone",
            "telephone number",
            "official website",
            "website",
            "address",
            "location",
            "where is lcwu",
            "where is the university",
            "general contact",
        ]

        if any(
            keyword in query_lower
            for keyword in general_info_keywords
        ):

            for chunk in self.chunks:

                chunk_lower = chunk.lower()

                if (
                    "official website:" in chunk_lower
                    and "general contact:" in chunk_lower
                ):
                    return [chunk]

        # ========================================================
        # QUOTA
        # ========================================================

        if (
            "quota" in query_lower
            or "reserved seats" in query_lower
            or "reserved quota" in query_lower
        ):

            for chunk in self.chunks:

                if chunk.strip().startswith("QUOTA"):

                    return [chunk]

        # ========================================================
        # SCHOLARSHIPS
        # ========================================================

        if (
            "scholarship" in query_lower
            or "scholarships" in query_lower
            or "financial aid" in query_lower
        ):

            for chunk in self.chunks:

                if chunk.strip().startswith(
                    "SCHOLARSHIPS AND FINANCIAL AID"
                ):

                    return [chunk]

        # ========================================================
        # HOSTEL
        # ========================================================

        if (
            "hostel" in query_lower
            or "hostels" in query_lower
            or "accommodation" in query_lower
        ):

            for chunk in self.chunks:

                if chunk.strip().startswith("HOSTEL"):

                    return [chunk]

        # ========================================================
        # ADMISSION PROCESS
        # ========================================================

        admission_process_keywords = [
            "admission process",
            "admission procedure",
            "how to apply",
            "how do i apply",
            "application process",
            "apply for admission",
        ]

        if any(
            keyword in query_lower
            for keyword in admission_process_keywords
        ):

            for chunk in self.chunks:

                if chunk.strip().startswith(
                    "ADMISSION PROCESS"
                ):

                    return [chunk]

        # ========================================================
        # ADMISSION REQUIREMENTS
        # ========================================================

        admission_requirement_keywords = [
            "admission requirement",
            "admission requirements",
            "requirements for admission",
            "requirements to apply",
            "eligibility",
            "eligibility criteria",
            "eligible",
            "what are the requirements",
            "what requirements",
        ]

        if any(
            keyword in query_lower
            for keyword in admission_requirement_keywords
        ):

            for chunk in self.chunks:

                if chunk.strip().startswith("ADMISSIONS"):

                    return [chunk]

        # ========================================================
        # GENERAL ADMISSION QUESTIONS
        # ========================================================

        if (
            "admission" in query_lower
            or "apply" in query_lower
            or "application" in query_lower
            or "merit" in query_lower
        ):

            for chunk in self.chunks:

                if chunk.strip().startswith("ADMISSIONS"):

                    return [chunk]

        # ========================================================
        # UNDERGRADUATE PROGRAMS
        # ========================================================

        if (
            "undergraduate program" in query_lower
            or "undergraduate programs" in query_lower
            or "degree program" in query_lower
            or "degree programs" in query_lower
            or "what programs" in query_lower
        ):

            for chunk in self.chunks:

                if chunk.strip().startswith(
                    "UNDERGRADUATE PROGRAMS"
                ):

                    return [chunk]

        # ========================================================
        # FACULTIES / DEPARTMENTS
        # ========================================================

        if (
            "faculty" in query_lower
            or "faculties" in query_lower
            or "department" in query_lower
            or "departments" in query_lower
        ):

            for chunk in self.chunks:

                if chunk.strip().startswith(
                    "FACULTIES AND DEPARTMENTS"
                ):

                    return [chunk]

        # Rank remaining sections with BM25-style lexical matching,
        # avoiding a heavyweight embedding model at application startup.
        stop_words = {
            "a", "an", "and", "are", "as", "at", "be", "by", "do",
            "for", "from", "how", "i", "in", "is", "it", "me", "of",
            "on", "or", "the", "to", "what", "when", "where", "which",
            "who", "with",
        }

        def tokenize(text):
            return re.findall(r"[a-z0-9]+", text.lower())

        query_terms = set(tokenize(query)) - stop_words

        if not query_terms or not self.chunks:
            return []

        documents = [
            Counter(tokenize(chunk))
            for chunk in self.chunks
        ]
        document_frequency = Counter(
            term
            for document in documents
            for term in document
        )
        average_length = sum(
            sum(document.values())
            for document in documents
        ) / len(documents)

        results = []

        for chunk, document in zip(self.chunks, documents):
            document_length = sum(document.values())
            score = 0.0

            for term in query_terms:
                frequency = document[term]

                if frequency:
                    inverse_frequency = math.log(
                        1
                        + (
                            len(documents)
                            - document_frequency[term]
                            + 0.5
                        )
                        / (document_frequency[term] + 0.5)
                    )
                    length_normalization = (
                        1.2
                        * (
                            0.25
                            + 0.75 * document_length / average_length
                        )
                    )
                    score += inverse_frequency * (
                        frequency * 2.2
                    ) / (frequency + length_normalization)

            if score > 0:
                results.append((score, chunk))

        results.sort(key=lambda result: result[0], reverse=True)
        return [chunk for _, chunk in results[:top_k]]

    # ============================================================
    # SOURCE INFORMATION
    # ============================================================

    def get_source(self, chunk):

        chunk_text = chunk.strip()

        if chunk_text.startswith("ADMISSIONS"):

            return {
                "title": "LCWU Admissions",
                "section": "Admissions",
                "page": "University Information"
            }

        if chunk_text.startswith(
            "UNDERGRADUATE PROGRAMS"
        ):

            return {
                "title": "LCWU Undergraduate Programs",
                "section": "Undergraduate Programs",
                "page": "University Information"
            }

        if chunk_text.startswith(
            "FACULTIES AND DEPARTMENTS"
        ):

            return {
                "title": "LCWU Faculties and Departments",
                "section": "Faculties and Departments",
                "page": "University Information"
            }

        if chunk_text.startswith(
            "ADMISSION PROCESS"
        ):

            return {
                "title": "LCWU Admission Process",
                "section": "Admission Process",
                "page": "University Information"
            }

        if chunk_text.startswith("ADMISSION OFFICE"):

            return {
                "title": "LCWU Admission Office",
                "section": "Admission Office",
                "page": "University Information"
            }

        if chunk_text.startswith("HOSTEL"):

            return {
                "title": "LCWU Hostel",
                "section": "Hostel",
                "page": "University Information"
            }

        if chunk_text.startswith(
            "SCHOLARSHIPS AND FINANCIAL AID"
        ):

            return {
                "title": "LCWU Scholarships and Financial Aid",
                "section": "Scholarships and Financial Aid",
                "page": "University Information"
            }

        if chunk_text.startswith("QUOTA"):

            return {
                "title": "LCWU Admission Quota",
                "section": "Quota",
                "page": "University Information"
            }

        return {
            "title": "LCWU University Information",
            "section": "General",
            "page": "University Information"
        }