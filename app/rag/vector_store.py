import faiss
from sentence_transformers import SentenceTransformer


class UniversityVectorStore:
    def __init__(self, chunks):
        self.chunks = chunks

        # Convert university text into embeddings
        self.model = SentenceTransformer("all-MiniLM-L6-v2")

        embeddings = self.model.encode(
            chunks,
            convert_to_numpy=True
        ).astype("float32")

        # Create FAISS vector index
        self.index = faiss.IndexFlatL2(embeddings.shape[1])
        self.index.add(embeddings)

    def search(self, query, top_k=1):

        query_lower = " ".join(query.lower().split())

        # ============================================================
        # DIRECT SECTION RETRIEVAL
        # ============================================================

        # Quota
        if "quota" in query_lower:
            for chunk in self.chunks:
                if chunk.strip().startswith("QUOTA"):
                    return [chunk]

        # Scholarships
        if "scholarship" in query_lower:
            for chunk in self.chunks:
                if chunk.strip().startswith(
                    "SCHOLARSHIPS AND FINANCIAL AID"
                ):
                    return [chunk]

        # Hostel
        if "hostel" in query_lower:
            for chunk in self.chunks:
                if chunk.strip().startswith("HOSTEL"):
                    return [chunk]

        # Admission process
        if (
            "admission process" in query_lower
            or "admission procedure" in query_lower
        ):
            for chunk in self.chunks:
                if chunk.strip().startswith("ADMISSION PROCESS"):
                    return [chunk]

        # Undergraduate programs
        if (
            "undergraduate program" in query_lower
            or "undergraduate programs" in query_lower
            or "degree programs" in query_lower
        ):
            for chunk in self.chunks:
                if chunk.strip().startswith("UNDERGRADUATE PROGRAMS"):
                    return [chunk]

        # Faculties and departments
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

        # ============================================================
        # GENERAL ADMISSION QUESTIONS
        # ============================================================

        if (
            "admission" in query_lower
            or "apply" in query_lower
            or "application" in query_lower
            or "eligibility" in query_lower
            or "merit" in query_lower
        ):
            admission_chunks = []

            for chunk in self.chunks:
                chunk_lower = chunk.lower()

                if (
                    "admissions" in chunk_lower
                    or "admission process" in chunk_lower
                ):
                    admission_chunks.append(chunk)

            if admission_chunks:
                return admission_chunks[:top_k]

        # ============================================================
        # FALLBACK: FAISS SEMANTIC SEARCH
        # ============================================================

        query_embedding = self.model.encode(
            [query],
            convert_to_numpy=True
        ).astype("float32")

        distances, indices = self.index.search(
            query_embedding,
            len(self.chunks)
        )

        results = []

        for distance, index in zip(
            distances[0],
            indices[0]
        ):
            if index < len(self.chunks):
                results.append(
                    (distance, self.chunks[index])
                )

        # Lower L2 distance = more similar
        results.sort(key=lambda item: item[0])

        return [
            chunk
            for _, chunk in results[:top_k]
        ]

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

        if chunk_text.startswith("UNDERGRADUATE PROGRAMS"):
            return {
                "title": "LCWU Undergraduate Programs",
                "section": "Undergraduate Programs",
                "page": "University Information"
            }

        if chunk_text.startswith("FACULTIES AND DEPARTMENTS"):
            return {
                "title": "LCWU Faculties and Departments",
                "section": "Faculties and Departments",
                "page": "University Information"
            }

        if chunk_text.startswith("ADMISSION PROCESS"):
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

        if chunk_text.startswith("SCHOLARSHIPS AND FINANCIAL AID"):
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