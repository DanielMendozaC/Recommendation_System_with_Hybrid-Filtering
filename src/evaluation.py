import numpy as np
from sklearn.metrics import mean_squared_error
import pandas as pd

def calculate_rmse(y_true, y_pred):
    """Calculate Root Mean Square Error"""
    return np.sqrt(mean_squared_error(y_true, y_pred))

def precision_at_k(recommended_items, relevant_items, k=10):
    """Calculate precision@k"""
    # Take only k recommendations
    recommended_items = recommended_items[:k]
    
    # Calculate precision
    relevant_and_recommended = len(set(recommended_items) & set(relevant_items))
    precision = relevant_and_recommended / len(recommended_items)
    
    return precision

def recall_at_k(recommended_items, relevant_items, k=10):
    """Calculate recall@k"""
    # Take only k recommendations
    recommended_items = recommended_items[:k]
    
    # Calculate recall
    relevant_and_recommended = len(set(recommended_items) & set(relevant_items))
    recall = relevant_and_recommended / len(relevant_items) if len(relevant_items) > 0 else 0
    
    return recall

def evaluate_model(model, test_data, user_item_matrix):
    """Evaluate model performance"""
    results = []
    
    # Group test data by user
    user_groups = test_data.groupby('user_id')
    
    for user_id, group in user_groups:
        # Skip users not in training data
        if user_id not in user_item_matrix.index:
            continue
        
        # Get relevant items for this user
        relevant_items = group['item_id'].tolist()
        
        # Generate recommendations
        try:
            recommended_items = model.recommend(user_id, n_recommendations=10)
            
            # Calculate metrics
            prec = precision_at_k(recommended_items, relevant_items, k=10)
            rec = recall_at_k(recommended_items, relevant_items, k=10)
            
            # Store results
            results.append({
                'user_id': user_id,
                'precision@10': prec,
                'recall@10': rec
            })
        except:
            continue
    
    # Calculate average metrics
    results_df = pd.DataFrame(results)
    avg_precision = results_df['precision@10'].mean()
    avg_recall = results_df['recall@10'].mean()
    
    return {
        'precision@10': avg_precision,
        'recall@10': avg_recall,
        'detailed_results': results_df
    }