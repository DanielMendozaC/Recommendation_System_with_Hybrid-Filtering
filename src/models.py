import numpy as np
from scipy.sparse.linalg import svds
from sklearn.metrics.pairwise import cosine_similarity

class CollaborativeFilteringRecommender:
    """Matrix Factorization based Collaborative Filtering"""
    
    def __init__(self, n_factors=50):
        self.n_factors = n_factors
        self.user_factors = None
        self.item_factors = None
        self.user_ids = None
        self.item_ids = None
        
    def fit(self, user_item_matrix):
        """Train the model using SVD"""
        self.user_ids = user_item_matrix.index
        self.item_ids = user_item_matrix.columns
        
        # Apply SVD
        U, sigma, Vt = svds(user_item_matrix.values, k=self.n_factors)
        
        # Convert sigma to diagonal matrix
        sigma_diag = np.diag(sigma)
        
        # Get latent factors
        self.user_factors = U
        self.item_factors = Vt.T
        
        return self
    
    def recommend(self, user_idx, n_recommendations=10):
        """Generate recommendations for a user"""
        # Get user's predicted ratings
        user_pred = self.user_factors[user_idx, :].dot(self.item_factors.T)
        
        # Get indices of top N recommendations
        recommendations = np.argsort(user_pred)[::-1][:n_recommendations]
        
        return recommendations

class ContentBasedRecommender:
    """Content-based recommender using TF-IDF vectors"""
    
    def __init__(self):
        self.item_features = None
        self.item_ids = None
        self.similarity_matrix = None
        
    def fit(self, item_features, item_ids):
        """Train the model by computing item similarities"""
        self.item_features = item_features
        self.item_ids = item_ids
        
        # Compute similarity matrix
        self.similarity_matrix = cosine_similarity(item_features)
        
        return self
    
    def recommend(self, item_idx, n_recommendations=10):
        """Find similar items to a given item"""
        # Get similarities to all other items
        item_similarities = self.similarity_matrix[item_idx]
        
        # Get indices of top N similar items (excluding itself)
        similar_indices = np.argsort(item_similarities)[::-1][1:n_recommendations+1]
        
        return similar_indices

class HybridRecommender:
    """Hybrid recommender combining collaborative and content-based approaches"""
    
    def __init__(self, cf_weight=0.7):
        self.cf_recommender = CollaborativeFilteringRecommender()
        self.cb_recommender = ContentBasedRecommender()
        self.cf_weight = cf_weight
        self.user_item_matrix = None
        self.user_to_items = None
        
    def fit(self, user_item_matrix, item_features, item_ids, train_data):
        """Train both models"""
        self.user_item_matrix = user_item_matrix
        self.cf_recommender.fit(user_item_matrix)
        self.cb_recommender.fit(item_features, item_ids)
        
        # Create mapping of users to items they've rated
        self.user_to_items = train_data.groupby('user_id')['item_id'].apply(list).to_dict()
        
        return self
    
    def recommend(self, user_id, n_recommendations=10):
        """Generate hybrid recommendations"""
        # Get user index
        user_idx = np.where(self.user_item_matrix.index == user_id)[0][0]
        
        # Get CF recommendations
        cf_rec_indices = self.cf_recommender.recommend(user_idx, n_recommendations=n_recommendations)
        cf_recommendations = self.user_item_matrix.columns[cf_rec_indices].tolist()
        
        # Get items the user has interacted with
        user_items = self.user_to_items.get(user_id, [])
        
        # For each item, get content-based recommendations
        cb_recommendations = []
        for item in user_items:
            if item in self.cb_recommender.item_ids:
                item_idx = np.where(self.cb_recommender.item_ids == item)[0][0]
                similar_indices = self.cb_recommender.recommend(item_idx, n_recommendations=3)
                cb_recommendations.extend(self.cb_recommender.item_ids[similar_indices])
        
        # Combine recommendations with weighting
        final_recommendations = []
        
        # Add CF recommendations with weight
        for item in cf_recommendations:
            final_recommendations.append((item, self.cf_weight))
            
        # Add CB recommendations with weight
        cb_weight = 1.0 - self.cf_weight
        for item in cb_recommendations:
            found = False
            for i, (rec_item, weight) in enumerate(final_recommendations):
                if rec_item == item:
                    final_recommendations[i] = (rec_item, weight + cb_weight)
                    found = True
                    break
            if not found:
                final_recommendations.append((item, cb_weight))
        
        # Sort by weight and take top N
        final_recommendations.sort(key=lambda x: x[1], reverse=True)
        final_recommendations = [item for item, _ in final_recommendations[:n_recommendations]]
        
        return final_recommendations