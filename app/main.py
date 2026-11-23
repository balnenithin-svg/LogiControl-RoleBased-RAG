
import os
from datetime import datetime, timedelta, timezone

import jwt
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Depends
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel, Field

from .rag import RoleBasedRAG

# Load local environment variables
load_dotenv()

# FastAPI application
app = FastAPI(
    title="LogiControl Role-Based RAG API",
    description="AI Logistics Knowledge Assistant",
    version="1.0.0"
)

# JWT configuration
SECRET_KEY = os.getenv("JWT_SECRET_KEY")
ALGORITHM = "HS256"
TOKEN_EXPIRY_HOURS = 2

if not SECRET_KEY:
    raise RuntimeError("JWT_SECRET_KEY is missing")

security = HTTPBearer()

# Initialize RAG system
rag = RoleBasedRAG()

# Role-based users
USERS = {
    "admin": {
        "password": os.getenv("ADMIN_PASSWORD"),
        "role": "admin"
    },
    "operations": {
        "password": os.getenv("OPERATIONS_PASSWORD"),
        "role": "operations"
    },
    "dispatcher": {
        "password": os.getenv("DISPATCHER_PASSWORD"),
        "role": "dispatcher"
    },
    "support": {
        "password": os.getenv("SUPPORT_PASSWORD"),
        "role": "customer_support"
    }
}


# Request models
class LoginRequest(BaseModel):
    username: str = Field(min_length=1)
    password: str = Field(min_length=1)


class ChatRequest(BaseModel):
    question: str = Field(min_length=1)


# Generate JWT token
def create_token(username: str, role: str):
    payload = {
        "sub": username,
        "role": role,
        "exp": datetime.now(timezone.utc)
        + timedelta(hours=TOKEN_EXPIRY_HOURS)
    }

    return jwt.encode(
        payload,
        SECRET_KEY,
        algorithm=ALGORITHM
    )


# Verify JWT token
def get_current_user(
    credentials: HTTPAuthorizationCredentials = Depends(security)
):
    try:
        payload = jwt.decode(
            credentials.credentials,
            SECRET_KEY,
            algorithms=[ALGORITHM]
        )

        username = payload.get("sub")
        role = payload.get("role")

        user = USERS.get(username)

        if not user or user["role"] != role:
            raise HTTPException(
                status_code=401,
                detail="Invalid user or role"
            )

        if not user["password"]:
            raise HTTPException(
                status_code=401,
                detail="User account is not configured"
            )

        return {
            "username": username,
            "role": role
        }

    except jwt.ExpiredSignatureError:
        raise HTTPException(
            status_code=401,
            detail="Token has expired. Please log in again."
        )

    except jwt.InvalidTokenError:
        raise HTTPException(
            status_code=401,
            detail="Invalid authentication token"
        )


# Health check
@app.get("/")
def home():
    return {
        "message": "LogiControl Role-Based RAG API is running",
        "status": "healthy"
    }


# Login endpoint
@app.post("/login")
def login(request: LoginRequest):

    username = request.username.strip().lower()
    user = USERS.get(username)

    if not user or not user["password"]:
        raise HTTPException(
            status_code=401,
            detail="Invalid credentials or account not configured"
        )

    if request.password != user["password"]:
        raise HTTPException(
            status_code=401,
            detail="Invalid credentials"
        )

    token = create_token(
        username,
        user["role"]
    )

    return {
        "access_token": token,
        "token_type": "bearer",
        "username": username,
        "role": user["role"],
        "expires_in": TOKEN_EXPIRY_HOURS * 3600
    }


# Role-based RAG chat endpoint
@app.post("/chat")
def chat(
    request: ChatRequest,
    current_user: dict = Depends(get_current_user)
):
    try:
        result = rag.ask(
            question=request.question,
            role=current_user["role"]
        )

        return {
            "username": current_user["username"],
            "role": current_user["role"],
            "answer": result["answer"],
            "sources": result["sources"]
        }

    except ValueError as error:
        raise HTTPException(
            status_code=400,
            detail=str(error)
        )

    except Exception as error:
        print("RAG error:", str(error))

        raise HTTPException(
            status_code=500,
            detail="Unable to process the RAG query"
        )
