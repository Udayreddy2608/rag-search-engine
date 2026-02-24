from PIL import Image
from sentence_transformers import SentenceTransformer


class MultimodalSearch:
    def __init__(self, model_name="clip-ViT-B-32"):
        self.model = SentenceTransformer(model_name)

    def embed_image(self, image_path):
        try:
            image = Image.open(image_path).convert("RGB")
            embeddings = self.model.encode([image])
            return embeddings[0]
        except Exception as e:
            print(f"Exception occurred: {e}")
            raise


def verify_image_embedding(image_path):
    try:
        mms = MultimodalSearch()
        embedding = mms.embed_image(image_path=image_path)
        print(embedding.shape)
    except Exception as e:
        print(f"Exception occurred: {e}")
        raise