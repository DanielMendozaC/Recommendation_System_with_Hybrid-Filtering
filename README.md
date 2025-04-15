# Amazon Product Recommendation System

An end-to-end recommendation system using the Amazon Electronics dataset to provide personalized product recommendations through hybrid collaborative and content-based filtering approaches.

## Overview

This project implements a comprehensive recommendation system that combines multiple approaches:

1. **Collaborative Filtering**: Matrix factorization using SVD to identify latent factors in user-product interactions
2. **Content-Based Filtering**: Using TF-IDF vectors of product reviews to find similar products
3. **Hybrid Approach**: Weighted combination of both methods for improved recommendation quality

## Performance

The hybrid approach demonstrates superior performance:

| Model | Precision@10 | Recall@10 |
|-------|-------------|-----------|
| Collaborative Filtering | 0.068 | 0.075 |
| Content-Based | 0.042 | 0.051 |
| **Hybrid** | **0.085** | **0.092** |

## Key Features

- Complete data processing pipeline
- Multiple recommendation algorithms
- Comprehensive evaluation framework
- Deployed API endpoint for real-time recommendations
- Clear visualization of model performance

## Technologies Used

- Python, Pandas, NumPy, scikit-learn
- Flask for API deployment
- Docker for containerization
- AWS for cloud deployment

## Getting Started

1. Clone the repository
2. Download the dataset from [Amazon Review Dataset](https://nijianmo.github.io/amazon/index.html)
3. Run the data preprocessing script
4. Train the models using the provided notebook
5. Deploy the API locally or to the cloud

## Future Improvements

- Deep learning-based recommendation models
- Incorporating more product metadata
- A/B testing framework
- Real-time model updates