import argparse
from search_utils import load_movies
from hybrid_search import HybridSearch
from gemini import image_description
from google.genai import types
import mimetypes

def main():
    parser = argparse.ArgumentParser(
        description="Image Description CLI"
    )

    parser.add_argument(
        "--image",
        type=str,
        required=True,
        help="Path of the image to be described"
    )

    parser.add_argument(
        "--query",
        type=str,
        required=True,
        help="User query"
    )

    args = parser.parse_args()

    mime, _ = mimetypes.guess_type(args.image)
    mime = mime or "image/jpeg"

    with open(args.image, "rb") as f:
        img_content = f.read()

    parts = [
        types.Part.from_bytes(
            data=img_content,
            mime_type=mime
        ),
        args.query.strip()
    ]

    response = image_description(parts=parts)

    print(response.text.strip())

    print(f"Rewritten query: {response.text.strip()}")
    if response.usage_metadata is not None:
        print(f"Total tokens:    {response.usage_metadata.total_token_count}")
if __name__ == "__main__":
    main()