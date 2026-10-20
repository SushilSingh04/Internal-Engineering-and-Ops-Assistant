"""
Unified Embeddings Provider with Cloud and Local 1536-Dimensional Support.

Supports:
  1. Cloud: Google Generative AI (gemini-embedding-001 / text-embedding-004) -> 1536 dims
  2. Local: Orange/orange-nomic-v1.5-1536 (SentenceTransformer) -> 1536 dims
  3. Auto detection & fallback:
     - If running on a deployed service (or ENVIRONMENT=production) -> Cloud Provider (Google)
     - If running locally (development / local machine) -> Local Provider
     - If primary provider fails -> Automatic fallback with warning logs
"""
from __future__ import annotations

import logging
import os
from typing import Any, List

import torch
from langchain_core.embeddings import Embeddings
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from transformers import PreTrainedModel

from config import (
    EMBEDDING_DIMENSIONS,
    EMBEDDING_MODEL,
    EMBEDDING_PROVIDER,
    ENVIRONMENT,
    GOOGLE_API_KEY,
    LOCAL_EMBEDDING_MODEL,
)

logger = logging.getLogger(__name__)

# Ensure NomicBertModel compatibility on newer transformers
if not hasattr(PreTrainedModel, "get_extended_attention_mask"):
    def _get_extended_attention_mask(self, attention_mask, input_shape):
        if attention_mask.dim() == 3:
            extended = attention_mask[:, None, :, :]
        elif attention_mask.dim() == 2:
            extended = attention_mask[:, None, None, :]
        else:
            raise ValueError(f"Wrong shape for attention_mask {attention_mask.shape}")
        extended = extended.to(dtype=self.dtype)
        return (1.0 - extended) * torch.finfo(self.dtype).min

    PreTrainedModel.get_extended_attention_mask = _get_extended_attention_mask


def is_deployed_service() -> bool:
    """Detect if running in a deployed cloud environment versus local machine."""
    # Explicit override via env
    if os.getenv("IS_LOCAL", "").lower() in ("true", "1", "yes"):
        return False
    if os.getenv("DEPLOYED", "").lower() in ("true", "1", "yes"):
        return True
    if ENVIRONMENT.lower() in ("production", "prod", "staging"):
        return True

    # Cloud hosting platform markers
    cloud_env_markers = [
        "RENDER",                   # Render.com
        "RAILWAY_ENVIRONMENT",      # Railway
        "FLY_APP_NAME",            # Fly.io
        "KUBERNETES_SERVICE_HOST",  # Kubernetes
        "AWS_EXECUTION_ENV",        # AWS Lambda / ECS
        "AWS_LAMBDA_FUNCTION_NAME",
        "GAE_APPLICATION",          # Google App Engine
        "GOOGLE_CLOUD_PROJECT",     # Google Cloud Run / Functions
        "DYNO",                     # Heroku
        "VERCEL",                   # Vercel
    ]
    return any(os.getenv(var) is not None for var in cloud_env_markers)


class Local1536Embeddings(Embeddings):
    """
    Local SentenceTransformer embeddings model producing exactly 1536-dimensional vectors.
    Uses 'Orange/orange-nomic-v1.5-1536' with Nomic prefixes for high retrieval accuracy.
    """

    def __init__(self, model_name: str = LOCAL_EMBEDDING_MODEL):
        self.model_name = model_name
        self._model = None

    @property
    def model(self):
        if self._model is None:
            logger.info("Loading local 1536-dim embedding model: %s ...", self.model_name)
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer(self.model_name, trust_remote_code=True)
            get_dim_fn = getattr(self._model, "get_embedding_dimension", self._model.get_sentence_embedding_dimension)
            dim = get_dim_fn()
            if dim != EMBEDDING_DIMENSIONS:
                logger.warning(
                    "Local embedding model dimension %d does not match expected %d",
                    dim,
                    EMBEDDING_DIMENSIONS,
                )
        return self._model

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        prefixed = [f"search_document: {t}" for t in texts]
        vectors = self.model.encode(prefixed, show_progress_bar=False, normalize_embeddings=True)
        return [v.tolist() for v in vectors]

    def embed_query(self, text: str) -> List[float]:
        prefixed = f"search_query: {text}"
        vector = self.model.encode(prefixed, show_progress_bar=False, normalize_embeddings=True)
        return vector.tolist()


class FallbackEmbeddings(Embeddings):
    """
    Tries primary embeddings first. If it fails, seamlessly falls back to secondary embeddings.
    """

    def __init__(self, primary: Embeddings, fallback: Embeddings, primary_name: str, fallback_name: str):
        self.primary = primary
        self.fallback = fallback
        self.primary_name = primary_name
        self.fallback_name = fallback_name

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        try:
            return self.primary.embed_documents(texts)
        except Exception as exc:
            logger.warning(
                "Primary embedding provider '%s' failed: %s. Falling back to '%s'.",
                self.primary_name,
                exc,
                self.fallback_name,
            )
            return self.fallback.embed_documents(texts)

    def embed_query(self, text: str) -> List[float]:
        try:
            return self.primary.embed_query(text)
        except Exception as exc:
            logger.warning(
                "Primary embedding provider '%s' failed: %s. Falling back to '%s'.",
                self.primary_name,
                exc,
                self.fallback_name,
            )
            return self.fallback.embed_query(text)


def get_embeddings() -> Embeddings:
    """
    Get configured embedding model based on environment and provider settings:
      - If running on a deployed service -> Cloud (Google GenAI) with Local fallback
      - If running locally -> Local (Orange/orange-nomic-v1.5-1536) with Cloud fallback
      - Explicit EMBEDDING_PROVIDER override: 'local', 'cloud', or 'auto'
    """
    cloud_embeddings = GoogleGenerativeAIEmbeddings(
        model=EMBEDDING_MODEL,
        google_api_key=GOOGLE_API_KEY,
        output_dimensionality=EMBEDDING_DIMENSIONS,
    )
    local_embeddings = Local1536Embeddings()

    provider = os.getenv("EMBEDDING_PROVIDER", EMBEDDING_PROVIDER).lower().strip()

    if provider == "local":
        logger.info("Embedding provider set to 'local' (1536 dims).")
        return FallbackEmbeddings(
            primary=local_embeddings,
            fallback=cloud_embeddings,
            primary_name="Local (1536-dim Nomic)",
            fallback_name="Cloud (Google GenAI)",
        )

    if provider == "cloud":
        logger.info("Embedding provider set to 'cloud' (Google GenAI 1536 dims).")
        return FallbackEmbeddings(
            primary=cloud_embeddings,
            fallback=local_embeddings,
            primary_name="Cloud (Google GenAI)",
            fallback_name="Local (1536-dim Nomic)",
        )

    # Auto detection based on environment
    deployed = is_deployed_service()
    if deployed:
        logger.info("Deployed service environment detected -> using Cloud provider (Google GenAI) with Local fallback.")
        return FallbackEmbeddings(
            primary=cloud_embeddings,
            fallback=local_embeddings,
            primary_name="Cloud (Google GenAI)",
            fallback_name="Local (1536-dim Nomic)",
        )
    else:
        logger.info("Local environment detected -> using Local 1536-dim provider with Cloud fallback.")
        return FallbackEmbeddings(
            primary=local_embeddings,
            fallback=cloud_embeddings,
            primary_name="Local (1536-dim Nomic)",
            fallback_name="Cloud (Google GenAI)",
        )
