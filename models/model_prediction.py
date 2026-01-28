import pickle
import pandas as pd
import os


# Absolute path of THIS file
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

rating_pivot = pickle.load(
    open(os.path.join(BASE_DIR, "rating_pivot.pkl"), "rb")
)

item_similarity = pickle.load(
    open(os.path.join(BASE_DIR, "item_similarity.pkl"), "rb")
)


df = pickle.load(
    open(os.path.join(BASE_DIR, "reviews_df.pkl"), "rb")
)

rf = pickle.load(
    open(os.path.join(BASE_DIR, "sentiment_model.pkl"), "rb")
)

tfidf = pickle.load(
    open(os.path.join(BASE_DIR, "tfidf.pkl"), "rb")
)



def recommend_item_based(user, n=20):
    if user not in rating_pivot.index:
        return []

    user_ratings = rating_pivot.loc[user]
    rated_items = user_ratings[user_ratings > 0]

    scores = {}

    for item, rating in rated_items.items():
        for sim_item, sim in item_similarity[item].items():
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

    