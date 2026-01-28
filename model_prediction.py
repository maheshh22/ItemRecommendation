import pickle
import pandas as pd


rating_pivot = pickle.load(open("rating_pivot.pkl", "rb"))
item_similarity_df = pickle.load(open("item_similarity.pkl", "rb"))
df = pickle.load(open("reviews_df.pkl", "rb"))
rf = pickle.load(open("sentiment_model.pkl", "rb"))
tfidf = pickle.load(open("tfidf.pkl", "rb"))

def recommend_item_based(user, n=20):
    if user not in rating_pivot.index:
        return []

    user_ratings = rating_pivot.loc[user]
    rated_items = user_ratings[user_ratings > 0]

    scores = {}

    for item, rating in rated_items.items():
        for sim_item, sim in item_similarity_df[item].items():
            if user_ratings[sim_item] == 0:
                scores[sim_item] = scores.get(sim_item, 0) + sim * rating

    
    return (
        pd.Series(scores)
        .sort_values(ascending=False)
        .head(n)
        .index
        .tolist()
    )

def get_sentiment_score(product):
    reviews = df[df['name'] == product]['clean_text']

    if reviews.empty:
        return 0.0

    probs = rf.predict_proba(tfidf.transform(reviews))

    # Case 1: binary probabilities [p_neg, p_pos]
    if probs.ndim == 2:
        return probs[:, 1].mean()

    # Case 2: only positive probability returned
    return probs.mean()

    