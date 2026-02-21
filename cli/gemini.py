import os
from dotenv import load_dotenv
from google import genai

api_key = os.environ.get("GEMINI_API_KEY")


def generate_response(query:str):
    try:
        client = genai.Client(api_key= api_key)
        prompt = f"""Fix any spelling errors in this movie search query.

                    Only correct obvious typos. Don't change correctly spelled words.

                    Query: "{query}"

                    If no errors, return the original query.
                Corrected:"""
        response = client.models.generate_content(model = "gemini-2.5-flash", contents = prompt)
        return response.text
    except:
        pass

def rewrite_query(query:str):
    try:
        client = genai.Client(api_key = api_key)
        prompt = f"""Fix any spelling errors in this movie search query.

                    Only correct obvious typos. Don't change correctly spelled words.

                    Query: "{query}"

                    If no errors, return the original query.
                Corrected:"""
        response = client.models.generate_content(model = "gemini-2.5-flash", contents = prompt)
        print("==========RESPONSE========")
        print(response.text)
        return response.text
    except:
        pass

def expand_query(query:str):
    try:
        client = genai.client(api_key = api_key)
        prompt = f"""Expand this movie search query with related terms.

        Add synonyms and related concepts that might appear in movie descriptions.
        Keep expansions relevant and focused.
        This will be appended to the original query.

        Examples:

        - "scary bear movie" -> "scary horror grizzly bear movie terrifying film"
        - "action movie with bear" -> "action thriller bear chase fight adventure"
        - "comedy with bear" -> "comedy funny bear humor lighthearted"

        Query: "{query}"
        """
        response = client.models.generate_contet(model = "gemini-2.5-flash", contents = prompt)
        return response.text
    except Exception as e:
        print(f"Error occured during expansion")