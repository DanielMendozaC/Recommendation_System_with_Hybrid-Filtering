import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
import pickle
import os

def load_data(file_path):
    """Load the Amazon Electronics dataset"""
    df = pd.read_json(file_path, lines=True)
    return df

def preprocess_data(df):
    """Preprocess the data for recommendation"""
    df = df[['reviewerID', 'asin', 'overall', 'reviewText']]
    
    # Rename columns for clarity
    df.columns = ['user_id', 'item_id', 'rating', 'review_text']
    
    # Filter out users with less than 5 ratings
    user_counts = df['user_id'].value_counts()
    df = df[df['user_id'].isin(user_counts[user_counts >= 5].index)]
    
    # Create train/test split
    train_data, test_data = train_test_split(df, test_size=0.2, random_state=42)
    
    return train_data, test_data

def create_user_item_matrix(df):
    """Create user-item interaction matrix"""
    user_item_matrix = df.pivot(index='user_id', columns='item_id', values='rating').fillna(0)
    return user_item_matrix

def create_item_features(df):
    """Create item features from review text"""
    # Aggregate reviews by item
    item_reviews = df.groupby('item_id')['review_text'].apply(' '.join).reset_index()
    
    # Create TF-IDF vectors
    vectorizer = TfidfVectorizer(max_features=5000, stop_words='english')
    item_features = vectorizer.fit_transform(item_reviews['review_text'])
    
    return item_features, vectorizer, item_reviews['item_id'].values

def save_processed_data(data, file_path):
    """Save processed data to disk"""
    with open(file_path, 'wb') as f:
        pickle.dump(data, f)