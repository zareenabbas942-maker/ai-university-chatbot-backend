import os

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.routes.chat import router as chat_router


configured_origins = os.getenv("CORS_ORIGINS", "")
allowed_origins = [
    "http://localhost:3000",
    "http://127.0.0.1:3000",
    *(
        origin.strip().rstrip("/")
        for origin in configured_origins.split(",")
        if origin.strip()
    ),
]

app = FastAPI(
    title="AI University Chatbot",
    description="AI-powered university information chatbot",
    version="1.0.0"
)


# ============================================================
# CORS CONFIGURATION
# ============================================================

app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_origin_regex=r"https://[a-zA-Z0-9-]+\.vercel\.app",
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================
# HOME ROUTE
# ============================================================

@app.get("/")
def home():
    return {
        "message": "AI University Chatbot Backend is running!"
    }


# ============================================================
# HEALTH CHECK
# ============================================================

@app.get("/api/health")
def health_check():
    return {
        "status": "healthy"
    }


# ============================================================
# CHAT ROUTES
# ============================================================

app.include_router(
    chat_router,
    prefix="/api"
)