"""Amazon Bedrock provider.

Uses the Anthropic Messages API format on Bedrock. Reads AWS creds from the
default boto3 chain (env vars, ~/.aws/credentials, IAM role, etc.).
"""

from __future__ import annotations

import json

import boto3

from ..config import settings


class BedrockClient:
    name = "bedrock"

    def __init__(self) -> None:
        self._client = boto3.client("bedrock-runtime", region_name=settings.aws_region)
        self._model_id = settings.bedrock_model_id

    def generate(self, *, system: str, user: str, max_tokens: int = 1024) -> str:
        body = {
            "anthropic_version": "bedrock-2023-05-31",
            "max_tokens": max_tokens,
            "system": system,
            "messages": [{"role": "user", "content": user}],
            "temperature": 0.0,
        }
        resp = self._client.invoke_model(
            modelId=self._model_id,
            body=json.dumps(body),
            contentType="application/json",
            accept="application/json",
        )
        data = json.loads(resp["body"].read())
        return data["content"][0]["text"]
