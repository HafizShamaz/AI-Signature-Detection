
import os
from pathlib import Path

import boto3
from botocore.config import Config


BUCKET_NAME = os.environ["STORAGE_BUCKET"]

s3 = boto3.client(
    "s3",
    endpoint_url=os.environ["AWS_ENDPOINT_URL_S3"],
    region_name=os.environ["AWS_REGION"],
    aws_access_key_id=os.environ["AWS_ACCESS_KEY_ID"],
    aws_secret_access_key=os.environ["AWS_SECRET_ACCESS_KEY"],
    config=Config(
        signature_version="s3v4",
        s3={"addressing_style": "path"},
        retries={"mode": "standard", "max_attempts": 3},
    ),
)


def upload_image(file_obj, object_key, content_type):
    file_obj.seek(0)

    s3.upload_fileobj(
        file_obj,
        BUCKET_NAME,
        object_key,
        ExtraArgs={"ContentType": content_type},
    )


def download_image(object_key, destination):
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)

    s3.download_file(
        BUCKET_NAME,
        object_key,
        str(destination),
    )


def delete_image(object_key):
    s3.delete_object(
        Bucket=BUCKET_NAME,
        Key=object_key,
    )
