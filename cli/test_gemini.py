import os
from dotenv import load_dotenv

load_dotenv()
api_key = os.environ.get("GEMINI_API_KEY")
print(f"Using key {api_key[:6]}...")

from google import genai

client = genai.Client(api_key=api_key)

response = client.models.generate_content(model = "gemini-2.5-flash", contents = "Which is greater 9.11 or 9.9?")

print(response.text)

met_dat = response.usage_metadata

print(f"Prompt Tokens: {met_dat.prompt_token_count}")
print(f"Response Tokens: {met_dat.candidates_token_count}")