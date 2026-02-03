import json

DATA_PATH = "data/movies.json"
STOPWORDS_PATH = "data/stopwords.txt"

def load_movies() -> list[dict]:
    with open(DATA_PATH,"r") as f:
        data = json.load(f)
    return data['movies']

def load_stop_words() -> list:
    with open(STOPWORDS_PATH,"r") as f:
        data = f.read()
        words = data.splitlines()
    return words