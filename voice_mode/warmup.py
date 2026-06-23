"""Warm up the local STT/TTS services at startup.

The first inference against whisper.cpp and Kokoro is slow: the model loads
into memory and, on Apple Silicon, CoreML/Metal compiles the graph on that
first run. Without warming, the user pays that cost (several seconds) on their
very first voice turn. Warming fires one throwaway request at each local
service when the server boots, so the slow first inference happens before
anyone is waiting on it.
"""

import asyncio
import io
import logging
import time
import wave

from openai import AsyncOpenAI

from .config import (
    TTS_BASE_URLS,
    STT_BASE_URLS,
    TTS_MODELS,
    STT_MODEL,
    KOKORO_DEFAULT_VOICE,
    WHISPER_LANGUAGE,
    OPENAI_API_KEY,
)
from .provider_discovery import is_local_provider

logger = logging.getLogger("voicemode")

# A dummy key keeps the OpenAI SDK happy when talking to a local server that
# does not check auth (matches the pattern in simple_failover.py).
_LOCAL_KEY = OPENAI_API_KEY or "dummy-key-for-local"


def _silent_wav(seconds: float = 0.3, rate: int = 16000) -> io.BytesIO:
    """Build a tiny silent mono WAV in memory for the STT warm-up request."""
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)  # 16-bit
        wf.setframerate(rate)
        wf.writeframes(b"\x00\x00" * int(rate * seconds))
    buf.seek(0)
    # The OpenAI SDK infers the upload content type from the file name.
    buf.name = "warmup.wav"
    return buf


async def _warm_tts(base_url: str) -> None:
    client = AsyncOpenAI(api_key=_LOCAL_KEY, base_url=base_url, timeout=30.0, max_retries=0)
    model = TTS_MODELS[0] if TTS_MODELS else "tts-1"
    await client.audio.speech.create(
        model=model,
        voice=KOKORO_DEFAULT_VOICE,
        input="ok",
        response_format="wav",
    )


async def _warm_stt(base_url: str) -> None:
    client = AsyncOpenAI(api_key=_LOCAL_KEY, base_url=base_url, timeout=30.0, max_retries=0)
    # whisper.cpp needs "auto" passed explicitly; its default is English.
    language = WHISPER_LANGUAGE or "auto"
    await client.audio.transcriptions.create(
        model=STT_MODEL,
        file=_silent_wav(),
        response_format="text",
        language=language,
    )


async def _warm_one(label: str, base_url: str, coro_fn) -> None:
    start = time.perf_counter()
    try:
        await coro_fn(base_url)
        elapsed = (time.perf_counter() - start) * 1000
        logger.info(f"Warm-up: {label} ready at {base_url} ({elapsed:.0f}ms)")
    except Exception as e:
        # Warm-up is best-effort. A failure here must never affect the server;
        # the real request path has its own failover and error handling.
        logger.debug(f"Warm-up: {label} at {base_url} failed (ignored): {e}")


async def warm_up_services() -> None:
    """Fire one throwaway request at each local STT and TTS endpoint."""
    tasks = []
    for base_url in TTS_BASE_URLS:
        if is_local_provider(base_url):
            tasks.append(_warm_one("TTS", base_url, _warm_tts))
    for base_url in STT_BASE_URLS:
        if is_local_provider(base_url):
            tasks.append(_warm_one("STT", base_url, _warm_stt))

    if not tasks:
        logger.debug("Warm-up: no local endpoints to warm")
        return

    logger.info(f"Warm-up: warming {len(tasks)} local voice endpoint(s)")
    await asyncio.gather(*tasks)


def warm_up_in_background() -> None:
    """Start warm-up in a daemon thread so it never blocks server startup."""
    import threading

    def _run() -> None:
        try:
            asyncio.run(warm_up_services())
        except Exception as e:
            logger.debug(f"Warm-up thread error (ignored): {e}")

    threading.Thread(target=_run, daemon=True, name="voicemode-warmup").start()
