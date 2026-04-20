#!/usr/bin/env python
# coding: utf-8
"""
Storage configuration module.

Manages S3/MinIO connection settings and environment variable fallbacks.
"""

import os
from dataclasses import dataclass
from typing import Optional


@dataclass
class S3Config:
    """Configuration for S3/MinIO storage."""

    endpoint_url: Optional[str]
    access_key_id: str
    secret_access_key: str
    bucket_name: str
    region_name: str = "us-east-1"
    use_s3: bool = True

    @classmethod
    def from_env(cls) -> "S3Config":
        """
        Create S3Config from environment variables.

        Environment Variables:
        ----------------------
        Endpoint (one of):
            - MINIO_ENDPOINT
            - MINIO_ENDPOINT_URL
            - S3_ENDPOINT_URL

        Access Key (one of):
            - MINIO_ACCESS_KEY
            - AWS_ACCESS_KEY_ID

        Secret Key (one of):
            - MINIO_SECRET_KEY
            - AWS_SECRET_ACCESS_KEY

        Bucket Name (one of):
            - ASSETS_BUCKET
            - S3_BUCKET_NAME

        Other:
            - AWS_REGION (default: us-east-1)
            - USE_S3 (default: true)

        Returns:
            S3Config: Configuration instance
        """
        # Endpoint fallbacks: MINIO_ENDPOINT, MINIO_ENDPOINT_URL, S3_ENDPOINT_URL
        endpoint = (
            os.getenv("MINIO_ENDPOINT")
            or os.getenv("MINIO_ENDPOINT_URL")
            or os.getenv("S3_ENDPOINT_URL")
            or None  # When unset, let s3fs/botocore pick AWS endpoint
        )

        # Access key fallbacks: MINIO_ACCESS_KEY, AWS_ACCESS_KEY_ID
        access_key = os.getenv("MINIO_ACCESS_KEY") or os.getenv("AWS_ACCESS_KEY_ID") or ""

        # Secret key fallbacks: MINIO_SECRET_KEY, AWS_SECRET_ACCESS_KEY
        secret_key = os.getenv("MINIO_SECRET_KEY") or os.getenv("AWS_SECRET_ACCESS_KEY") or ""

        # Bucket name fallbacks: ASSETS_BUCKET, S3_BUCKET_NAME
        bucket_name = os.getenv("ASSETS_BUCKET") or os.getenv("S3_BUCKET_NAME") or "chatbot-assets"

        # Region
        region_name = os.getenv("AWS_REGION", "us-east-1")

        # Use S3 only when explicitly enabled. This keeps local/dev installs
        # from accidentally attempting AWS with invalid default credentials.
        use_s3 = os.getenv("USE_S3", "false").lower() == "true"

        return cls(
            endpoint_url=endpoint,
            access_key_id=access_key,
            secret_access_key=secret_key,
            bucket_name=bucket_name,
            region_name=region_name,
            use_s3=use_s3,
        )

    def to_storage_options(self) -> dict:
        """
        Convert config to fsspec/s3fs storage options.

        Returns:
            dict: Storage options for fsspec.open()
        """
        opts = {
            "key": self.access_key_id,
            "secret": self.secret_access_key,
            "config_kwargs": {"s3": {"addressing_style": "path"}},
        }

        if self.endpoint_url:
            opts["client_kwargs"] = {"endpoint_url": self.endpoint_url}

        return opts


def get_s3_config() -> S3Config:
    """
    Get S3 configuration from environment variables.

    This is evaluated at call time (not import time) so that notebooks can
    call load_dotenv() after importing modules and still have up-to-date
    credentials and endpoint settings.

    Returns:
        S3Config: Current S3 configuration
    """
    return S3Config.from_env()


def is_s3_enabled() -> bool:
    """
    Check if S3 is enabled based on configuration.

    Returns:
        bool: True if S3 should be used
    """
    config = get_s3_config()
    return config.use_s3 and all(
        [
            config.access_key_id,
            config.secret_access_key,
            config.bucket_name,
        ]
    )
