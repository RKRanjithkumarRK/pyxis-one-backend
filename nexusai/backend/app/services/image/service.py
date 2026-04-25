"""Image generation service — Flux Pro/Schnell, SDXL, DALL-E 3, Imagen 3."""
from __future__ import annotations
import asyncio
import base64
import logging
import uuid
from typing import Any

logger = logging.getLogger("nexusai.image.service")

# Supported models and their backends
MODELS: dict[str, dict] = {
    "flux-pro": {"backend": "replicate", "model": "black-forest-labs/flux-pro"},
    "flux-schnell": {"backend": "replicate", "model": "black-forest-labs/flux-schnell"},
    "sdxl": {"backend": "replicate", "model": "stability-ai/sdxl:39ed52f2319f9f..."},
    "dall-e-3": {"backend": "openai", "model": "dall-e-3"},
    "imagen-3": {"backend": "google", "model": "imagen-3.0-generate-001"},
}

DEFAULT_MODEL = "flux-schnell"


async def generate_image(
    prompt: str,
    model: str = DEFAULT_MODEL,
    width: int = 1024,
    height: int = 1024,
    num_images: int = 1,
    negative_prompt: str = "",
    style: str | None = None,
) -> list[str]:
    """Generate images and return list of base64-encoded data URIs or URLs."""
    cfg = MODELS.get(model) or MODELS[DEFAULT_MODEL]
    backend = cfg["backend"]

    tasks = [
        _generate_one(prompt, cfg, width, height, negative_prompt, style)
        for _ in range(min(num_images, 4))
    ]
    results = await asyncio.gather(*tasks, return_exceptions=True)
    urls = []
    for r in results:
        if isinstance(r, Exception):
            logger.warning("Image generation failed: %s", r)
        else:
            urls.append(r)
    return urls


async def _generate_one(
    prompt: str,
    cfg: dict,
    width: int,
    height: int,
    negative_prompt: str,
    style: str | None,
) -> str:
    backend = cfg["backend"]
    if backend == "replicate":
        return await _replicate(prompt, cfg["model"], width, height, negative_prompt)
    elif backend == "openai":
        return await _openai(prompt, width, height, style)
    elif backend == "google":
        return await _google(prompt, width, height)
    raise ValueError(f"Unknown backend: {backend}")


async def _replicate(prompt: str, model: str, width: int, height: int, negative_prompt: str) -> str:
    import replicate
    output = await asyncio.to_thread(
        replicate.run,
        model,
        input={
            "prompt": prompt,
            "width": width,
            "height": height,
            "negative_prompt": negative_prompt,
            "num_outputs": 1,
        },
    )
    url = output[0] if isinstance(output, list) else output
    return str(url)


async def _openai(prompt: str, width: int, height: int, style: str | None) -> str:
    import openai
    size_map = {(1024, 1024): "1024x1024", (1792, 1024): "1792x1024", (1024, 1792): "1024x1792"}
    size = size_map.get((width, height), "1024x1024")
    client = openai.AsyncOpenAI()
    resp = await client.images.generate(
        model="dall-e-3",
        prompt=prompt,
        n=1,
        size=size,
        style=style or "vivid",
        response_format="url",
    )
    return resp.data[0].url


async def _google(prompt: str, width: int, height: int) -> str:
    import google.generativeai as genai
    model = genai.ImageGenerationModel("imagen-3.0-generate-001")
    result = await asyncio.to_thread(
        model.generate_images,
        prompt=prompt,
        number_of_images=1,
        aspect_ratio="1:1",
    )
    image = result.images[0]
    data = base64.b64encode(image._image_bytes).decode()
    return f"data:image/png;base64,{data}"


async def upscale_image(image_url: str, scale: int = 4) -> str:
    import replicate
    output = await asyncio.to_thread(
        replicate.run,
        "nightmareai/real-esrgan:42fed1c4...",
        input={"image": image_url, "scale": scale},
    )
    return str(output)


async def remove_background(image_url: str) -> str:
    """Remove background using rembg."""
    import httpx
    from rembg import remove
    from PIL import Image
    import io

    async with httpx.AsyncClient() as client:
        r = await client.get(image_url, timeout=30)
        r.raise_for_status()
        raw = r.content

    img = Image.open(io.BytesIO(raw)).convert("RGBA")
    out = await asyncio.to_thread(remove, img)
    buf = io.BytesIO()
    out.save(buf, format="PNG")
    data = base64.b64encode(buf.getvalue()).decode()
    return f"data:image/png;base64,{data}"
