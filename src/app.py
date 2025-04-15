from flask import Flask, request, jsonify
import pickle
import pandas as pd
import numpy as np

app = Flask(__name__)

# Load models and data
with open('models/hybrid_recommender.pkl', 'rb') as f:
    model = pickle.load(f)

with open('data/processed/product_metadata.pkl', 'rb') as f:
    product_metadata = pickle.load(f)

@app.route('/recommend', methods=['POST'])
def recommend():
    """Endpoint to get recommendations"""
    data = request.json
    user_id = data.get('user_id')
    n_recommendations = data.get('n_recommendations', 10)
    
    if user_id is None:
        return jsonify({'error': 'user_id is required'}), 400
    
    try:
        # Get recommendations
        recommendations = model.recommend(user_id, n_recommendations=n_recommendations)
        
        # Enrich with product metadata
        enriched_recommendations = []
        for item_id in recommendations:
            item_data = product_metadata.get(item_id, {})
            enriched_recommendations.append({
                'item_id': item_id,
                'title': item_data.get('title', 'Unknown'),
                'category': item_data.get('category', 'Unknown'),
                'price': item_data.get('price', 0.0)
            })
        
        return jsonify({'recommendations': enriched_recommendations})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)