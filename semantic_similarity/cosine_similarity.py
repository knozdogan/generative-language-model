import os
import pickle
from numpy import dot
from numpy.linalg import norm


CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))

with open(CURRENT_DIR + "/word_to_vector_trsf.pkl", "rb") as pk:
    word_to_vector = pickle.load(pk)

def cosine_similarity(vec_a, vec_b):
    return dot(vec_a,vec_b)

def similar_words(word="tree", top_k=10):
    return sorted(
        word_to_vector.keys(), 
        key=lambda x: -cosine_similarity(word_to_vector[x], word_to_vector[word])
    )[:top_k]

if __name__ == '__main__':
    print(similar_words())
