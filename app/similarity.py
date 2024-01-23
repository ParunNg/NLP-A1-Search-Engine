import numpy as np

def cosine_similarity(A, B):
    dot_product = np.dot(A, B)
    norm_a = np.linalg.norm(A)
    norm_b = np.linalg.norm(B)
    similarity = dot_product / (norm_a * norm_b) * 100
    return "{:.4f}".format(similarity)

# function to find the most similar word to the input vector
def get_most_similar(word, embeddings, n):
    # retrieve all words in our embeddings vocabs
    vocabs = list(embeddings.keys())
    names = ['word', 'sim']

    try:
        vector = embeddings[word.lower()]
    except:
        vector = embeddings['<UNK>']
    
    similarities = {}

    # for each word in the vocabs, find the cosine similarities between word vectors in our embeddings and the input vector
    for vocab in vocabs:
        similarities[vocab] = cosine_similarity(vector, embeddings[vocab])

    top_n = sorted(similarities.items(), key=lambda item: item[1], reverse=True)[1:n+1]

    return [dict(zip(names, val)) for val in top_n]