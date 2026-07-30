from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List

import httpx
import os
import tempfile

import torch
import torch.nn.functional as F

from speechbrain.inference.speaker import EncoderClassifier

app = FastAPI()

# Load SpeechBrain model once when the server starts
classifier = EncoderClassifier.from_hparams(
    source="speechbrain/spkrec-ecapa-voxceleb",
    savedir="pretrained_models/spkrec-ecapa-voxceleb",
)


class VerifyRequest(BaseModel):
    enrollment_urls: List[str]
    test_url: str


async def download_audio(url: str) -> str:
    """
    Downloads an audio file and returns the local temporary path.
    """

    async with httpx.AsyncClient(timeout=60.0) as client:
        response = await client.get(url)
        response.raise_for_status()

    extension = os.path.splitext(url)[1]
    if extension == "":
        extension = ".wav"

    tmp = tempfile.NamedTemporaryFile(
        delete=False,
        suffix=extension,
    )

    tmp.write(response.content)
    tmp.close()

    return tmp.name


def extract_embedding(audio_path: str):
    """
    Extract a speaker embedding from an audio file.
    """

    embedding = classifier.encode_file(audio_path)

    return embedding.squeeze()


@app.post("/verify-speaker")
async def verify_speaker(request: VerifyRequest):

    if len(request.enrollment_urls) == 0:
        raise HTTPException(
            status_code=400,
            detail="At least one enrollment audio is required.",
        )

    enrollment_embeddings = []

    # Download and process enrollment recordings
    for url in request.enrollment_urls:

        path = await download_audio(url)

        try:
            embedding = extract_embedding(path)
            enrollment_embeddings.append(embedding)

        finally:
            os.remove(path)

    # Average all enrollment embeddings
    target_profile = torch.mean(
        torch.stack(enrollment_embeddings),
        dim=0,
    )

    # Download test recording
    test_path = await download_audio(request.test_url)

    try:
        test_embedding = extract_embedding(test_path)

    finally:
        os.remove(test_path)

    # Cosine similarity
    similarity = F.cosine_similarity(
        target_profile.unsqueeze(0),
        test_embedding.unsqueeze(0),
    ).item()

    # Simple threshold
    threshold = 0.65

    return {
        "similarity": similarity,
        "verified": similarity >= threshold,
        "threshold": threshold,
    }