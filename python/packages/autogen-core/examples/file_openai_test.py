"""
Practical test of the File class with OpenAI.
"""

import os
import base64
from pathlib import Path

# Import our File class
from autogen_core import File

# Import OpenAI for testing with real API
from openai import OpenAI

# Check for API key
api_key = os.environ.get("OPENAI_API_KEY")
if not api_key:
    print("Please set the OPENAI_API_KEY environment variable")
    print("Example: export OPENAI_API_KEY=your_api_key_here")
    exit(1)

# Create a simple PDF file for testing
test_pdf_path = Path("test_document.pdf")
if not test_pdf_path.exists():
    print(f"Creating sample PDF file at {test_pdf_path}")
    with open(test_pdf_path, "wb") as f:
        # This is not a real PDF, just content for testing
        f.write(b"This is a test PDF document for File class testing.")

# Load the file using our File class
pdf_file = File.from_file(test_pdf_path)
print(f"Loaded file: {pdf_file.filename} ({pdf_file.mime_type})")

# Create the format required by OpenAI
openai_content = [
    pdf_file.to_openai_format(),
    {
        "type": "text",
        "text": "What's in this document?"
    }
]

# Print the content we're sending (for debug)
print("\nSending to OpenAI:")
print(f"File: {pdf_file.filename}")
print(f"Question: What's in this document?")

# Initialize OpenAI client
client = OpenAI()

# Make the API call
print("\nSending request to OpenAI API...")
try:
    completion = client.chat.completions.create(
        model="gpt-4o",  # Use gpt-4o as it supports file inputs
        messages=[
            {
                "role": "user",
                "content": openai_content
            }
        ]
    )
    
    # Print the response
    print("\nResponse from OpenAI:")
    print(completion.choices[0].message.content)
    
    print("\nTest successful! The File class works with OpenAI API.")
except Exception as e:
    print(f"\nError: {e}")
    print("\nNote: This test requires GPT-4o or a model that supports file inputs.")
    print("If you don't have access to these models, the test will fail.")

# Clean up the test file
if test_pdf_path.exists():
    test_pdf_path.unlink()
    print(f"\nCleaned up test file: {test_pdf_path}")
