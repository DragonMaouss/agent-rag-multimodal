# src/auth.py (Extrait de la fonction verify_password)
import logging
from datetime import datetime, timedelta
from typing import List, Optional
import bcrypt
import jwt
from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer
from pydantic import BaseModel

from src.config import settings

logger = logging.getLogger(__name__)

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="login")

class User(BaseModel):
    username: str
    roles: List[str]
    disabled: Optional[bool] = False

class UserInDB(User):
    hashed_password: str

USERS_DB = {
    "admin": UserInDB(
        username="admin",
        hashed_password=settings.admin_password_hash,
        roles=["admin", "tech", "rh", "public"]
    ),
    "tech_user": UserInDB(
        username="tech_user",
        hashed_password=settings.tech_password_hash,
        roles=["tech", "public"]
    ),
    "rh_user": UserInDB(
        username="rh_user",
        hashed_password=settings.rh_password_hash,
        roles=["rh", "public"]
    ),
    "guest": UserInDB(
        username="guest",
        hashed_password=settings.guest_password_hash,
        roles=["public"]
    )
}

def verify_password(plain_password: str, hashed_password: str) -> bool:
    if not hashed_password:
        return False
    try:
        return bcrypt.checkpw(
            plain_password.encode('utf-8'),
            hashed_password.encode('utf-8')
        )
    except Exception as e:
        logger.error("Erreur de vérification du mot de passe : %s", e)
        return False

def create_access_token(data: dict, expires_delta: Optional[timedelta] = None) -> str:
    to_encode = data.copy()
    expire = datetime.utcnow() + (expires_delta or timedelta(minutes=settings.access_token_expire_minutes))
    to_encode.update({"exp": expire})
    return jwt.encode(to_encode, settings.jwt_secret_key, algorithm=settings.jwt_algorithm)

async def get_current_user(token: str = Depends(oauth2_scheme)) -> User:
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Identifiants non valides ou jeton expiré",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload = jwt.decode(token, settings.jwt_secret_key, algorithms=[settings.jwt_algorithm])
        username: str = payload.get("sub")
        roles: List[str] = payload.get("roles", ["public"])
        if username is None:
            raise credentials_exception
    except jwt.PyJWTError:
        raise credentials_exception

    user = USERS_DB.get(username)
    if user is None or user.disabled:
        raise credentials_exception
    return User(username=user.username, roles=roles)