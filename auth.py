import os
import httpx
from fastapi import Header, HTTPException

class UserStore:
    def __init__(self, token: str, user_id: str):
        self.token, self.user_id = token, user_id
        self.url = os.environ["SUPABASE_URL"].rstrip("/")
        self.key = os.environ["SUPABASE_PUBLISHABLE_KEY"]

    async def rows(self, table: str, query: str = ""):
        async with httpx.AsyncClient(timeout=30) as client:
            r = await client.get(f"{self.url}/rest/v1/{table}?user_id=eq.{self.user_id}&{query}", headers={"apikey": self.key, "Authorization": f"Bearer {self.token}"})
        if not r.is_success:
            raise HTTPException(502, "Research storage unavailable")
        return r.json()

async def require_user(authorization: str = Header(default="")):
    if not authorization.startswith("Bearer "):
        raise HTTPException(401, "Authentication required")
    token = authorization[7:]
    url, key = os.environ.get("SUPABASE_URL"), os.environ.get("SUPABASE_PUBLISHABLE_KEY")
    if not url or not key:
        raise HTTPException(503, "Authentication service not configured")
    async with httpx.AsyncClient(timeout=15) as client:
        r = await client.get(f"{url.rstrip('/')}/auth/v1/user", headers={"apikey": key, "Authorization": f"Bearer {token}"})
    if not r.is_success or not r.json().get("id"):
        raise HTTPException(401, "Invalid session")
    return UserStore(token, r.json()["id"])