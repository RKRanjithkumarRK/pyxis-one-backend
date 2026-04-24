from __future__ import annotations
import httpx
from dataclasses import dataclass
from app.core.config import settings


@dataclass
class OAuthUserInfo:
    provider_id: str
    email: str | None
    name: str | None
    avatar_url: str | None


async def exchange_google_code(code: str, redirect_uri: str) -> OAuthUserInfo:
    async with httpx.AsyncClient() as client:
        token_resp = await client.post(
            "https://oauth2.googleapis.com/token",
            data={
                "code": code,
                "client_id": settings.GOOGLE_CLIENT_ID,
                "client_secret": settings.GOOGLE_CLIENT_SECRET,
                "redirect_uri": redirect_uri,
                "grant_type": "authorization_code",
            },
        )
        token_resp.raise_for_status()
        tokens = token_resp.json()

        user_resp = await client.get(
            "https://www.googleapis.com/oauth2/v2/userinfo",
            headers={"Authorization": f"Bearer {tokens['access_token']}"},
        )
        user_resp.raise_for_status()
        info = user_resp.json()

    return OAuthUserInfo(
        provider_id=info["id"],
        email=info.get("email"),
        name=info.get("name"),
        avatar_url=info.get("picture"),
    )


async def exchange_github_code(code: str, redirect_uri: str) -> OAuthUserInfo:
    async with httpx.AsyncClient() as client:
        token_resp = await client.post(
            "https://github.com/login/oauth/access_token",
            data={
                "client_id": settings.GITHUB_CLIENT_ID,
                "client_secret": settings.GITHUB_CLIENT_SECRET,
                "code": code,
                "redirect_uri": redirect_uri,
            },
            headers={"Accept": "application/json"},
        )
        token_resp.raise_for_status()
        tokens = token_resp.json()

        headers = {
            "Authorization": f"Bearer {tokens['access_token']}",
            "Accept": "application/json",
        }
        user_resp = await client.get("https://api.github.com/user", headers=headers)
        user_resp.raise_for_status()
        info = user_resp.json()

        # GitHub may hide email; fetch primary verified email separately
        email: str | None = info.get("email")
        if not email:
            emails_resp = await client.get("https://api.github.com/user/emails", headers=headers)
            if emails_resp.is_success:
                for e in emails_resp.json():
                    if e.get("primary") and e.get("verified"):
                        email = e["email"]
                        break

    return OAuthUserInfo(
        provider_id=str(info["id"]),
        email=email,
        name=info.get("name") or info.get("login"),
        avatar_url=info.get("avatar_url"),
    )


OAUTH_HANDLERS = {
    "google": exchange_google_code,
    "github": exchange_github_code,
}
