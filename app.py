from flask import Flask, request, jsonify ,render_template
from models.model_prediction import recommend_item_based ,get_sentiment_score
import pandas as pd

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/recommend', methods=['GET'])
def recommend():
    user = request.args.get('user')

    products = recommend_item_based(user)
    scored = {p: get_sentiment_score(p) for p in products}
    #scored = {"fruit":4.2 ,"apple":3,"banana":3,"cucumber":5,"maize":3,"pede":4.4}

    final = (
        pd.Series(scored)
        .sort_values(ascending=False)
        .head(5)
        .index
        .tolist()
    )
    
    return jsonify(final)

if __name__ == "__main__":
    app.run(debug=True)
