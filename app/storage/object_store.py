"""
MinIO/S3-compatible object storage abstraction.
Handles file uploads, downloads, and bucket management.
"""
from typing import Optional, BinaryIO
from pathlib import Path
from minio import Minio
from minio.error import S3Error
from app.utils.config import settings
from app.utils.logging import get_logger
from app.utils.exceptions import StorageException

logger = get_logger(__name__)


class ObjectStore:
    """
    Object storage abstraction using MinIO (S3-compatible).
    Manages datasets, models, checkpoints, and artifacts.
    """
    
    def __init__(self):
        self.client = Minio(
            settings.minio_endpoint,
            access_key=settings.minio_access_key,
            secret_key=settings.minio_secret_key,
            secure=settings.minio_secure
        )
        
        # Bucket names
        self.buckets = {
            'datasets': settings.minio_bucket_datasets,
            'models': settings.minio_bucket_models,
            'checkpoints': settings.minio_bucket_checkpoints,
            'artifacts': settings.minio_bucket_artifacts
        }
        
        self._ensure_buckets()
        logger.info("Object store initialized", endpoint=settings.minio_endpoint)
    
    def _ensure_buckets(self):
        """Ensure all required buckets exist."""
        for bucket_type, bucket_name in self.buckets.items():
            try:
                if not self.client.bucket_exists(bucket_name):
                    self.client.make_bucket(bucket_name)
                    logger.info("Bucket created", bucket=bucket_name, type=bucket_type)
                else:
                    logger.debug("Bucket exists", bucket=bucket_name)
            except S3Error as e:
                logger.error("Failed to ensure bucket", bucket=bucket_name, error=str(e))
                raise StorageException(f"Failed to create bucket {bucket_name}: {str(e)}")
    
    def upload_file(
        self,
        file_path: str,
        object_name: str,
        bucket_type: str = 'datasets',
        content_type: str = 'application/octet-stream'
    ) -> str:
        """
        Upload a file to object storage.
        
        Args:
            file_path: Local file path
            object_name: Object name in bucket
            bucket_type: Bucket type ('datasets', 'models', 'checkpoints', 'artifacts')
            content_type: MIME type
        
        Returns:
            Object path (bucket/object_name)
        """
        bucket_name = self.buckets.get(bucket_type)
        if not bucket_name:
            raise StorageException(f"Invalid bucket type: {bucket_type}")
        
        try:
            self.client.fput_object(
                bucket_name,
                object_name,
                file_path,
                content_type=content_type
            )
            
            object_path = f"{bucket_name}/{object_name}"
            logger.info(
                "File uploaded",
                file_path=file_path,
                object_path=object_path,
                bucket_type=bucket_type
            )
            
            return f"s3://{object_path}"
        
        except S3Error as e:
            logger.error("Upload failed", file_path=file_path, error=str(e))
            raise StorageException(f"Failed to upload file: {str(e)}")
    
    def download_file(
        self,
        object_name: str,
        file_path: str,
        bucket_type: str = 'datasets'
    ):
        """
        Download a file from object storage.
        
        Args:
            object_name: Object name in bucket
            file_path: Local destination path
            bucket_type: Bucket type
        """
        bucket_name = self.buckets.get(bucket_type)
        if not bucket_name:
            raise StorageException(f"Invalid bucket type: {bucket_type}")
        
        try:
            self.client.fget_object(
                bucket_name,
                object_name,
                file_path
            )
            
            logger.info(
                "File downloaded",
                object_name=object_name,
                file_path=file_path,
                bucket_type=bucket_type
            )
        
        except S3Error as e:
            logger.error("Download failed", object_name=object_name, error=str(e))
            raise StorageException(f"Failed to download file: {str(e)}")
    
    def upload_fileobj(
        self,
        file_obj: BinaryIO,
        object_name: str,
        length: int,
        bucket_type: str = 'datasets',
        content_type: str = 'application/octet-stream'
    ) -> str:
        """
        Upload a file object to storage.
        
        Args:
            file_obj: File-like object
            object_name: Object name in bucket
            length: Content length
            bucket_type: Bucket type
            content_type: MIME type
        
        Returns:
            Object path
        """
        bucket_name = self.buckets.get(bucket_type)
        if not bucket_name:
            raise StorageException(f"Invalid bucket type: {bucket_type}")
        
        try:
            self.client.put_object(
                bucket_name,
                object_name,
                file_obj,
                length,
                content_type=content_type
            )
            
            object_path = f"{bucket_name}/{object_name}"
            logger.info("File object uploaded", object_path=object_path)
            
            return f"s3://{object_path}"
        
        except S3Error as e:
            logger.error("Upload failed", object_name=object_name, error=str(e))
            raise StorageException(f"Failed to upload file object: {str(e)}")
    
    def get_presigned_url(
        self,
        object_name: str,
        bucket_type: str = 'datasets',
        expires_seconds: int = 3600
    ) -> str:
        """
        Generate a presigned URL for downloading.
        
        Args:
            object_name: Object name in bucket
            bucket_type: Bucket type
            expires_seconds: URL expiration time
        
        Returns:
            Presigned URL
        """
        bucket_name = self.buckets.get(bucket_type)
        if not bucket_name:
            raise StorageException(f"Invalid bucket type: {bucket_type}")
        
        try:
            from datetime import timedelta
            url = self.client.presigned_get_object(
                bucket_name,
                object_name,
                expires=timedelta(seconds=expires_seconds)
            )
            
            logger.info("Presigned URL generated", object_name=object_name)
            return url
        
        except S3Error as e:
            logger.error("Failed to generate presigned URL", error=str(e))
            raise StorageException(f"Failed to generate presigned URL: {str(e)}")
    
    def delete_object(self, object_name: str, bucket_type: str = 'datasets'):
        """Delete an object from storage."""
        bucket_name = self.buckets.get(bucket_type)
        if not bucket_name:
            raise StorageException(f"Invalid bucket type: {bucket_type}")
        
        try:
            self.client.remove_object(bucket_name, object_name)
            logger.info("Object deleted", object_name=object_name, bucket_type=bucket_type)
        
        except S3Error as e:
            logger.error("Delete failed", object_name=object_name, error=str(e))
            raise StorageException(f"Failed to delete object: {str(e)}")
    
    def list_objects(self, prefix: str = "", bucket_type: str = 'datasets') -> list:
        """List objects in a bucket with optional prefix."""
        bucket_name = self.buckets.get(bucket_type)
        if not bucket_name:
            raise StorageException(f"Invalid bucket type: {bucket_type}")
        
        try:
            objects = self.client.list_objects(bucket_name, prefix=prefix)
            return [obj.object_name for obj in objects]
        
        except S3Error as e:
            logger.error("List failed", bucket_type=bucket_type, error=str(e))
            raise StorageException(f"Failed to list objects: {str(e)}")


# Global object store instance
object_store = ObjectStore()
