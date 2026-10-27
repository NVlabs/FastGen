# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import io
import os
from contextlib import contextmanager
from typing import Generator, Union

import boto3
from torch.distributed.checkpoint import FileSystemReader, FileSystemWriter
from torch.distributed.checkpoint.filesystem import FileSystemBase

import fastgen.utils.s3 as s3


_COPY_OBJECT_MAX_BYTES = 5 * 1024**3


class _DeferredCloseBytesIO(io.BytesIO):
    """Keep an S3 context-owned upload buffer readable until context exit."""

    def close(self) -> None:
        # PyTorch's filesystem writer closes the stream it is given. The S3
        # context manager still has to rewind and upload that stream afterwards.
        pass

    def release(self) -> None:
        super().close()


class S3FileSystem(FileSystemBase):
    """Implementation of FileSystem for AWS S3 storage, authenticated through boto3's default credential chain."""

    def __init__(self) -> None:
        self.s3_client = boto3.client("s3")

    @contextmanager
    def create_stream(self, path: Union[str, os.PathLike], mode: str) -> Generator[io.IOBase, None, None]:
        """
        Create a stream to read from or write to S3.

        Args:
            path: S3 URI in the format s3://bucket-name/key
            mode: 'rb' for reading, 'wb' for writing
        """
        bucket, key = s3.parse_uri(str(path))

        if mode == "rb":
            # For reading, download to memory stream
            stream = io.BytesIO()
            try:
                self.s3_client.download_fileobj(bucket, key, stream)
                stream.seek(0)
                yield stream
            finally:
                stream.close()
        elif mode == "wb":
            # For writing, use memory stream then upload
            stream = _DeferredCloseBytesIO()
            try:
                yield stream
                stream.seek(0)
                self.s3_client.upload_fileobj(stream, bucket, key)
            finally:
                stream.release()
        else:
            raise ValueError(f"Unsupported mode: {mode}")

    def concat_path(self, path: Union[str, os.PathLike], suffix: str) -> Union[str, os.PathLike]:
        """Concatenate S3 path with suffix."""
        path_str = str(path)
        if path_str.endswith("/"):
            return f"{path_str}{suffix}"
        return f"{path_str}/{suffix}"

    def rename(self, path: Union[str, os.PathLike], new_path: Union[str, os.PathLike]) -> None:
        """
        Rename (or move) an object in S3.

        In S3, this is implemented as a copy followed by a deletion of the original.
        """
        src_bucket, src_key = s3.parse_uri(str(path))
        dst_bucket, dst_key = s3.parse_uri(str(new_path))

        # Avoid boto3's managed transfer helper: some S3-compatible endpoints
        # return ETags it rejects after the server-side copy already succeeded.
        source_metadata = self.s3_client.head_object(Bucket=src_bucket, Key=src_key)
        source_size = source_metadata["ContentLength"]
        # botocore percent-encodes the dict form exactly once; a pre-quoted string would be encoded twice.
        copy_source = {"Bucket": src_bucket, "Key": src_key}

        if source_size <= _COPY_OBJECT_MAX_BYTES:
            self.s3_client.copy_object(Bucket=dst_bucket, Key=dst_key, CopySource=copy_source)
        else:
            # CopyObject is limited to 5 GiB. Boto3's managed helper performs
            # a multipart server-side copy for larger objects.
            self.s3_client.copy(copy_source, dst_bucket, dst_key)

        # Delete the original
        self.s3_client.delete_object(Bucket=src_bucket, Key=src_key)

    def init_path(self, path: Union[str, os.PathLike]) -> Union[str, os.PathLike]:
        """Initialize and validate S3 path."""
        path_str = str(path)
        if not s3.is_uri(path_str):
            raise ValueError(f"Invalid S3 URI: {path_str}. It must start with 's3://'")
        return path_str

    def mkdir(self, path: Union[str, os.PathLike]) -> None:
        """
        Create a "directory" in S3.

        Note: S3 doesn't have real directories, but we can create an empty object
        with a trailing slash to simulate a directory.
        """
        path_str = str(path)
        if not path_str.endswith("/"):
            path_str += "/"

        bucket, key = s3.parse_uri(path_str, allow_bucket_root=True)
        if key:  # Don't create empty object if this is just a bucket
            self.s3_client.put_object(Bucket=bucket, Key=key)

    @classmethod
    def validate_checkpoint_id(cls, checkpoint_id: Union[str, os.PathLike]) -> bool:
        """Validate if the checkpoint_id is a valid S3 URI."""
        checkpoint_id_str = str(checkpoint_id)
        try:
            if not s3.is_uri(checkpoint_id_str):
                return False
            s3.parse_uri(checkpoint_id_str)
            return True
        except ValueError:
            return False

    def exists(self, path: Union[str, os.PathLike]) -> bool:
        """Check if an object exists in S3."""
        return s3.exists(str(path))

    def rm_file(self, path: Union[str, os.PathLike]) -> None:
        """Remove a file from S3."""
        bucket, key = s3.parse_uri(str(path))
        self.s3_client.delete_object(Bucket=bucket, Key=key)


class S3StorageWriter(FileSystemWriter):
    def __init__(self, path: str, **kwargs) -> None:
        """
        Initialize an S3 writer for distributed checkpointing.

        Args:
            path (str): The S3 URI to write checkpoints to.
            kwargs (dict): Keyword arguments to pass to the parent :class:`FileSystemWriter`.
        """
        super().__init__(path=path, sync_files=False, **kwargs)
        self.fs = S3FileSystem()
        self.path = self.fs.init_path(path)

    @classmethod
    def validate_checkpoint_id(cls, checkpoint_id: Union[str, os.PathLike]) -> bool:
        return S3FileSystem.validate_checkpoint_id(checkpoint_id)


class S3StorageReader(FileSystemReader):
    def __init__(self, path: Union[str, os.PathLike]) -> None:
        """
        Initialize an S3 reader for distributed checkpointing.

        Args:
            path (Union[str, os.PathLike]): The S3 URI to read checkpoints from.
        """
        super().__init__(path=path)
        self.fs = S3FileSystem()
        self.path = self.fs.init_path(path)

    @classmethod
    def validate_checkpoint_id(cls, checkpoint_id: Union[str, os.PathLike]) -> bool:
        return S3FileSystem.validate_checkpoint_id(checkpoint_id)
