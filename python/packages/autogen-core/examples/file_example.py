#!/usr/bin/env python3
"""
Basic example showing how to use the File class
"""

from pathlib import Path

from autogen_core import File
from autogen_core.models import UserMessage

# Create a temporary PDF file for demonstration
temp_pdf_path = Path("example.pdf")
with open(temp_pdf_path, "wb") as f:
    f.write(b"This is a test PDF file for demonstration purposes.")

try:
    # Load the PDF file
    pdf_file = File.from_file(temp_pdf_path)
    print(f"Created file: {pdf_file.filename} ({pdf_file.mime_type})")
    print(f"Content length: {len(pdf_file.data)} bytes")
    print(f"Base64 (sample): {pdf_file.to_base64()[:30]}...\n")
    
    # Create a message with the file
    message = UserMessage(
        content=["Please analyze this PDF:", pdf_file],
        source="user"
    )
    print("Successfully created UserMessage with File!")
    print(f"Message type: {message.__class__.__name__}")
    print(f"Message content types: {[type(item).__name__ for item in message.content]}\n")
    
    # Show how to convert to OpenAI format
    print("OpenAI format for file:")
    openai_format = pdf_file.to_openai_format()
    print(f"Type: {openai_format['type']}")
    print(f"Filename: {openai_format['file']['filename']}")
    data_uri = openai_format['file']['file_data']
    print(f"Data URI (sample): {data_uri[:30]}...\n")
    
    # Example usage with OpenAI API
    print("Example usage with OpenAI API:\n")
    print("""from openai import OpenAI

client = OpenAI()

# Create content array with text and file
content = [
    {"type": "text", "text": "Please analyze this PDF:"},
    file_obj.to_openai_format()
]

# Make API call
response = client.chat.completions.create(
    model="gpt-4o",  # Or any model that supports file inputs
    messages=[{"role": "user", "content": content}]
)

print(response.choices[0].message.content)""")

finally:
    # Clean up
    if temp_pdf_path.exists():
        temp_pdf_path.unlink()
