from __future__ import annotations

import base64
import mimetypes
import re
from pathlib import Path
from typing import Any, Dict, Optional, cast

from pydantic import GetCoreSchemaHandler, ValidationInfo
from pydantic_core import core_schema
from typing_extensions import Literal


class File:
    """Represents a file that can be sent to an LLM.

    This class is designed to handle file attachments in messages to language models
    that support file inputs, such as GPT-4 Vision or GPT-4.1. It supports various file
    formats including PDF, text files, and any other file type.

    Example:

        Loading a file from disk:

        .. code-block:: python

            from autogen_core import File
            
            # Load a PDF file
            pdf_file = File.from_file(Path("document.pdf"))
            
            # Use in a message
            message = UserMessage(
                content=[
                    "Please analyze this document", 
                    pdf_file
                ],
                source="user"
            )

        Creating a File from bytes:

        .. code-block:: python

            # From raw bytes
            file_content = b"Document content"
            file_obj = File.from_bytes(file_content, "document.txt", "text/plain")

        Creating a File from base64:

        .. code-block:: python

            # From base64 string (e.g., when receiving data from an API)
            base64_content = "VGVzdCBjb250ZW50"  # "Test content" in base64
            file_obj = File.from_base64(base64_content, "document.txt")
    """

    def __init__(self, filename: str, data: bytes, mime_type: Optional[str] = None):
        self.filename = filename
        self.data = data
        self.mime_type = mime_type or self._guess_mime_type(filename)

    @staticmethod
    def _guess_mime_type(filename: str) -> str:
        """Guess the MIME type from the filename."""
        guessed_type = mimetypes.guess_type(filename)[0]
        return guessed_type or "application/octet-stream"

    @classmethod
    def from_file(cls, file_path: Path) -> File:
        """Create a File object from a file path."""
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        
        return cls(
            filename=file_path.name,
            data=file_path.read_bytes(),
            mime_type=mimetypes.guess_type(file_path.name)[0]
        )

    @classmethod
    def from_bytes(cls, data: bytes, filename: str, mime_type: Optional[str] = None) -> File:
        """Create a File object from bytes."""
        return cls(
            filename=filename,
            data=data,
            mime_type=mime_type
        )

    @classmethod
    def from_base64(cls, base64_str: str, filename: str, mime_type: Optional[str] = None) -> File:
        """Create a File object from a base64 encoded string."""
        data = base64.b64decode(base64_str)
        return cls(
            filename=filename,
            data=data,
            mime_type=mime_type
        )

    @classmethod
    def from_data_uri(cls, uri: str, filename: str) -> File:
        """Create a File object from a data URI."""
        if not re.match(r"data:[^;]+;base64,", uri):
            raise ValueError("Invalid URI format. It should be a base64 encoded data URI.")

        # Extract the MIME type and base64 data
        mime_match = re.match(r"data:([^;]+);base64,", uri)
        if not mime_match:
            raise ValueError("Could not extract MIME type from data URI")
            
        mime_type = mime_match.group(1)
        base64_data = re.sub(r"data:[^;]+;base64,", "", uri)
        
        return cls.from_base64(base64_data, filename, mime_type)

    def to_base64(self) -> str:
        """Convert the file data to a base64 encoded string."""
        return base64.b64encode(self.data).decode("utf-8")

    def to_data_uri(self) -> str:
        """Convert the file to a data URI."""
        base64_str = self.to_base64()
        return f"data:{self.mime_type};base64,{base64_str}"

    # Returns a format suitable for OpenAI API
    def to_openai_format(self) -> Dict[str, Any]:
        """Convert the file to the format expected by OpenAI API."""
        return {
            "type": "file",
            "file": {
                "filename": self.filename,
                "file_data": self.to_data_uri(),
            }
        }

    @classmethod
    def __get_pydantic_core_schema__(cls, source_type: Any, handler: GetCoreSchemaHandler) -> core_schema.CoreSchema:
        # Custom validation
        def validate(value: Any, validation_info: ValidationInfo) -> Any:  # Return Any instead of File
            if isinstance(value, dict):
                filename = cast(str | None, value.get("filename"))
                data = cast(str | None, value.get("data"))
                mime_type = cast(str | None, value.get("mime_type"))
                
                if filename is None or data is None:
                    raise ValueError("Expected 'filename' and 'data' keys in the dictionary")
                
                return cls.from_base64(data, filename, mime_type)
                
            elif isinstance(value, cls):
                return value
            elif hasattr(value, "__class__") and value.__class__.__name__ == "Image":
                # Allow Image objects to pass through validation
                return value
            else:
                # If we're validating an item in a list for UserMessage.content,
                # we need to allow Image objects to pass through
                module_name = getattr(value.__class__, "__module__", "")
                if module_name == "autogen_core._image" and getattr(value.__class__, "__name__", "") == "Image":
                    return value
                raise TypeError(f"Expected dict or {cls.__name__} instance, got {type(value)}")

        # Custom serialization
        def serialize(value: Any) -> dict[str, Any]:
            # Handle both File and Image objects
            if isinstance(value, cls):
                return {
                    "filename": value.filename,
                    "data": value.to_base64(),
                    "mime_type": value.mime_type
                }
            # For Image objects, delegate to their own serialization
            elif hasattr(value, "to_base64"):
                return {"data": value.to_base64()}
            return {"data": "", "filename": "unknown", "mime_type": "application/octet-stream"}

        return core_schema.with_info_after_validator_function(
            validate,
            core_schema.any_schema(),
            serialization=core_schema.plain_serializer_function_ser_schema(serialize),
        )
