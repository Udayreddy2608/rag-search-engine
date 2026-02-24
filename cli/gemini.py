import os
from dotenv import load_dotenv
from google import genai

load_dotenv()
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
        client = genai.Client(api_key = api_key)
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
        response = client.models.generate_content(model = "gemini-2.5-flash", contents = prompt)
        return response.text
    except Exception as e:
        print(f"Error occured during expansion: {e}")

def ranker(query, doc):
    try:
        client = genai.Client(api_key= api_key)
        prompt = f"""Rate how well this movie matches the search query.

                Query: "{query}"
                Movie: {doc.get("title", "")} - {doc.get("document", "")}

                Consider:
                - Direct relevance to query
                - User intent (what they're looking for)
                - Content appropriateness

                Rate 0-10 (10 = perfect match).
                Give me ONLY the number in your response, no other text or explanation.

                Score:"""
        response = client.models.generate_content(model = "gemini-2.5-flash", contents = prompt)
        return response.text
    
    except Exception as e:
        print(f"Error occured {e}")

def batch_rerank(query:str, doc_list_str):
    try:
        client = genai.Client(api_key= api_key)
        prompt =f"""Rank these movies by relevance to the search query.

            Query: "{query}"

            Movies:
            {doc_list_str}

            Return ONLY the IDs in order of relevance (best match first). Return a valid JSON list, nothing else. For example:

            [75, 12, 34, 2, 1]
            """
        response = client.models.generate_content(model="gemini-2.5-flash", contents= prompt)
        text = response.text.strip()
        if text.startswith("```json\n"):
            text = text[8:]
        elif text.startswith("```json"):
            text = text[7:]
        elif text.startswith("```\n"):
            text = text[4:]
        elif text.startswith("```"):
            text = text[3:]
        if text.endswith("```"):
            text = text[:-3]
        text = text.strip()
        return text
    except Exception as e:
        print(f"Exception occured {e}")

def parse_llm_scores(text, expected_len):
    import json, re

    match = re.search(r"\[[^\]]*\]", text)
    if not match:
        raise ValueError("No JSON array found in LLM response")

    scores = json.loads(match.group())

    if not isinstance(scores, list):
        raise ValueError("LLM output is not a list")

    if len(scores) != expected_len:
        raise ValueError("Mismatch in number of scores")

    if not all(isinstance(x, int) and 0 <= x <= 3 for x in scores):
        raise ValueError("Scores must be integers 0–3")

    return scores

def rate_results(query, formatted_results):
    try:
        client = genai.Client(api_key= api_key)
        import json
        results_str = [json.dumps(r, default=str) if isinstance(r, dict) else str(r) for r in formatted_results]
        prompt = f"""Rate how relevant each result is to this query on a 0-3 scale:
                    Query: "{query}"
                    Results:
                    {chr(10).join(results_str)}
                    Scale:
                    - 3: Highly relevant
                    - 2: Relevant
                    - 1: Marginally relevant
                    - 0: Not relevant
                    Do NOT give any numbers out than 0, 1, 2, or 3.
                    Return ONLY the scores in the same order you were given the documents. Return a valid JSON list, nothing else. For example:
                    [2, 0, 3, 2, 0, 1]"""
        response = client.models.generate_content(model="gemini-2.5-flash", contents = prompt)
        print("============ RESPONSE =========")
        print(response.text)
        return parse_llm_scores(text= response.text, expected_len=len(formatted_results))
    except Exception as e:
        print(f"Error in rate_results: {e}")
        return [0] * len(formatted_results)
    

def rag_generation(query, results):
    try:
        client = genai.Client(api_key= api_key)
        prompt = prompt = f"""Answer the question or provide information based on the provided documents. This should be tailored to Hoopla users. Hoopla is a movie streaming service.

            Query: {query}

            Documents:
            {results}

            Provide a comprehensive answer that addresses the query:"""
        response = client.models.generate_content(model= "gemini-2.5-flash", contents= prompt)
        return response.text
    except Exception as e:
        print(f"Exception occured {e}")
        raise

def summarization(query, results):
    try:
        client = genai.Client(api_key= api_key)
        prompt = f"""
        Provide information useful to this query by synthesizing information from multiple search results in detail.
        The goal is to provide comprehensive information so that users know what their options are.
        Your response should be information-dense and concise, with several key pieces of information about the genre, plot, etc. of each movie.
        This should be tailored to Hoopla users. Hoopla is a movie streaming service.
        Query: {query}
        Search Results:
        {results}
        Provide a comprehensive 3–4 sentence answer that combines information from multiple sources:
        """
        print("Entered Summarization")
        response = client.models.generate_content(contents= prompt, model = "gemini-2.5-flash")
        return response.text
    except Exception as e:
        print(f"Exception occured {e}")
        raise
        
def summarization_with_citations(query, documents):
    try:
        client = genai.Client(api_key= api_key)
        prompt = prompt = f"""Answer the question or provide information based on the provided documents.

            This should be tailored to Hoopla users. Hoopla is a movie streaming service.

            If not enough information is available to give a good answer, say so but give as good of an answer as you can while citing the sources you have.

            Query: {query}

            Documents:
            {documents}

            Instructions:
            - Provide a comprehensive answer that addresses the query
            - Cite sources using [1], [2], etc. format when referencing information
            - If sources disagree, mention the different viewpoints
            - If the answer isn't in the documents, say "I don't have enough information"
            - Be direct and informative

            Answer:"""
        summary = client.models.generate_content(model= "gemini-2.5-flash", contents= prompt)
        return summary.text
    except Exception as e:
        print(f"Exception occured {e}")
        raise

def question_answer(question, results):
    try:
        client = genai.Client(api_key= api_key)
        prompt = f"""Answer the user's question based on the provided movies that are available on Hoopla.

            This should be tailored to Hoopla users. Hoopla is a movie streaming service.

            Question: {question}

            Documents:
            {results}

            Instructions:
            - Answer questions directly and concisely
            - Be casual and conversational
            - Don't be cringe or hype-y
            - Talk like a normal person would in a chat conversation

            Answer:"""
        
        answer = client.models.generate_content(model= "gemini-2.5-flash", contents= prompt)
        return answer.text
    
    except Exception as e:
        print(f"Exception occured {e}")
        raise

def image_description(parts):
    try:
        client = genai.Client(api_key= api_key)
        prompt = """Given the included image and text query, rewrite the text query to improve search results from a movie database. Make sure to:
            - Synthesize visual and textual information
            - Focus on movie-specific details (actors, scenes, style, etc.)
            - Return only the rewritten query, without any additional commentary
            """
        parts.append(prompt)
        result = client.models.generate_content(model= "gemini-2.5-flash", contents=parts)
        return result
    except Exception as e:
        print(f"Exception occured: {e}")
        raise
