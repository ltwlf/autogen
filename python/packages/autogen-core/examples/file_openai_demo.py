#!/usr/bin/env python3
"""
Demo showing how to use the File class with OpenAI API
"""

import os
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
    
    # Create a message with the file
    message = UserMessage(
        content=["Please analyze this PDF:", pdf_file],
        source="user"
    )
    print("\nSuccessfully created UserMessage with File!")
    print(f"Message type: {message.__class__.__name__}")
    print(f"Message content types: {[type(item).__name__ for item in message.content]}")
    
    # Show how to convert to OpenAI format
    print("\nOpenAI format for file:")
    openai_format = pdf_file.to_openai_format()
    print(f"Type: {openai_format['type']}")
    print(f"Filename: {openai_format['file']['filename']}")
    data_uri = openai_format['file']['file_data']
    print(f"Data URI (sample): {data_uri[:30]}...")
    
    # Example usage with OpenAI API
    print("\nExample OpenAI API usage code:")
    print("""
from openai import OpenAI

# Need API key in the environment
client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

# Create content array with text and file
content = [
    {"type": "text", "text": "Please analyze this PDF:"},
    pdf_file.to_openai_format()
]

# Make the API call
completion = client.chat.completions.create(
    model="gpt-4o",  # Or another model that supports file inputs
    messages=[{"role": "user", "content": content}]
)

print(completion.choices[0].message.content)
""")

    # Actually make the call if OPENAI_API_KEY is set
    api_key = os.environ.get("OPENAI_API_KEY")
    if api_key:
        try:
            from openai import OpenAI
            
            client = OpenAI(api_key=api_key)
            
            content = [
                {"type": "text", "text": "Please analyze this PDF:"},
                pdf_file.to_openai_format()
            ]
            
            print("\nMaking actual OpenAI API call...")
            completion = client.chat.completions.create(
                model="gpt-4o",  # Or another model that supports file inputs
                messages=[{"role": "user", "content": content}]
            )
            
            print("\nResponse from OpenAI:")
            print(completion.choices[0].message.content)
        except ImportError:
            print("\nOpenAI package not installed. Install with: pip install openai")
        except Exception as e:
            print(f"\nError calling OpenAI API: {e}")
    else:
        print("\nNo OPENAI_API_KEY environment variable found. Skipping API call.")

finally:
    # Clean up
    if temp_pdf_path.exists():
        temp_pdf_path.unlink()
