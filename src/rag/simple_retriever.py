from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


def retrieve_relevant_documents(query, documents, top_k=3):
    if not documents:
        return []

    texts = [doc["content"] for doc in documents]

    vectorizer = TfidfVectorizer(stop_words="english")

    tfidf_matrix = vectorizer.fit_transform(texts)

    query_vector = vectorizer.transform([query])

    similarities = cosine_similarity(query_vector, tfidf_matrix)[0]

    ranked_indices = similarities.argsort()[::-1][:top_k]

    results = []

    for index in ranked_indices:
        results.append({
            "filename": documents[index]["filename"],
            "path": documents[index]["path"],
            "content": documents[index]["content"],
            "score": float(similarities[index])
        })

    return results