
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
    version="1.2.0"
)

# JWT configuration for existing Railway login
SECRET_KEY = os.getenv("JWT_SECRET_KEY")
ALGORITHM = "HS256"
TOKEN_EXPIRY_HOURS = 2

if not SECRET_KEY:
    raise RuntimeError("JWT_SECRET_KEY is missing")

security = HTTPBearer()

# Initialize RAG system
rag = RoleBasedRAG()

# Supabase client initialized inside SupabaseRAG
supabase = rag.supabase_rag.supabase

# Role-based users for existing Railway login
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

# Approved application roles
ALLOWED_ROLES = {
    "admin",
    "operations",
    "dispatcher",
    "customer_support"
}


# Request models
class LoginRequest(BaseModel):
    username: str = Field(min_length=1)
    password: str = Field(min_length=1)


class ChatRequest(BaseModel):
    question: str = Field(min_length=1, max_length=1000)


# Generate Railway JWT token
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


# Verify existing Railway JWT token
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


# Existing Railway login endpoint
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


# Existing Railway JWT-based RAG chat endpoint
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


# Supabase-authenticated RAG chat endpoint
@app.post("/chat/supabase")
def chat_with_supabase_auth(
    request: ChatRequest,
    credentials: HTTPAuthorizationCredentials = Depends(security)
):
    # Step 1: Verify Supabase access token
    try:
        auth_response = supabase.auth.get_user(
            credentials.credentials
        )
        auth_user = auth_response.user

    except Exception as error:
        print("Supabase authentication error:", str(error))
        raise HTTPException(
            status_code=401,
            detail="Invalid or expired Supabase session"
        )

    if not auth_user:
        raise HTTPException(
            status_code=401,
            detail="Supabase user could not be verified"
        )

    # Step 2: Check authenticated user metadata
    app_metadata = getattr(auth_user, "app_metadata", {}) or {}

    if str(app_metadata.get("status", "")).lower() == "inactive":
        raise HTTPException(
            status_code=403,
            detail="This account has been deactivated"
        )

    banned_until = getattr(auth_user, "banned_until", None)

    if banned_until:
        try:
            banned_time = datetime.fromisoformat(
                str(banned_until).replace("Z", "+00:00")
            )

            if banned_time.tzinfo is None:
                banned_time = banned_time.replace(
                    tzinfo=timezone.utc
                )

            if banned_time > datetime.now(timezone.utc):
                raise HTTPException(
                    status_code=403,
                    detail="This account is currently suspended"
                )

        except ValueError:
            raise HTTPException(
                status_code=401,
                detail="Unable to validate account suspension status"
            )

    # Step 3: Retrieve authoritative role from Supabase profiles
    try:
        profile_response = (
            supabase
            .table("profiles")
            .select("role,status")
            .eq("id", str(auth_user.id))
            .maybe_single()
            .execute()
        )

        profile = profile_response.data

    except Exception as error:
        print("Supabase profile lookup error:", str(error))
        raise HTTPException(
            status_code=503,
            detail="Unable to verify user profile"
        )

    if not profile:
        raise HTTPException(
            status_code=403,
            detail="User profile not found"
        )

    # Step 4: Validate actual database role and account status
    role = profile.get("role")
    status = str(profile.get("status") or "").lower()

    if role not in ALLOWED_ROLES:
        raise HTTPException(
            status_code=403,
            detail="Invalid or unauthorized user role"
        )

    if status != "active":
        raise HTTPException(
            status_code=403,
            detail="This account is not active"
        )

    # Step 5: Execute role-filtered Supabase RAG
    try:
        result = rag.ask(
            question=request.question,
            role=role
        )

        return {
            "username": auth_user.email or str(auth_user.id),
            "role": role,
            "answer": result["answer"],
            "sources": result["sources"],
            "source_details": result.get("source_details", [])
        }

    except ValueError as error:
        raise HTTPException(
            status_code=400,
            detail=str(error)
        )

    except Exception as error:
        print("Supabase RAG processing error:", str(error))
        raise HTTPException(
            status_code=500,
            detail="Unable to process the Supabase RAG query"
        )
