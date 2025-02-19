# Collaborative and Content-Based Product Recommender

This repository contains an implementation of a **Hybrid Product Recommender System** combining two recommendation techniques:
- **Collaborative Filtering** with Matrix Factorization (Truncated SVD)
- **Content-Based Filtering** using TF-IDF and Cosine Similarity

The system is designed to recommend products to users based on their preferences and product similarities.

## Table of Contents
- [Overview](#overview)
- [Technologies Used](#technologies-used)
- [Installation](#installation)

## Overview
The **Hybrid Product Recommender** uses collaborative filtering and content-based filtering to generate personalized recommendations for users.

1. **Collaborative Filtering**: We use **Matrix Factorization** (via **Truncated SVD**) to extract latent features from user-item interactions. This helps predict missing ratings and recommend products that users are likely to enjoy based on patterns in the ratings.
   
2. **Content-Based Filtering**: This method uses product attributes such as descriptions and tags to recommend similar products to those the user has shown interest in. It utilizes **TF-IDF** for feature extraction and **Cosine Similarity** for measuring similarity between products.

## Technologies Used
- Python
- `pandas` for data manipulation
- `scipy` for sparse matrices
- `sklearn` for machine learning models and similarity measures
- `numpy` for numerical operations

## Installation

To use this repository, follow the steps below to clone and install the necessary dependencies:

1. Clone this repository:
   ```bash
   git clone https://github.com/yourusername/collaborative-and-content-based-product-recommender.git
   cd collaborative-and-content-based-product-recommender
