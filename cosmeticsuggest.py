"""CosmeticSuggest.ipynb

### Step 1: Data Exploration and Preprocessing

- **Load the Datasets**: Use pandas to load both the `User_review_data.xlsx` and `Makeup_Products_Metadata.xlsx` files.
- **Explore the Data**: Understand the structure of the datasets, including missing values, data types, and statistics of numerical columns.
- **Preprocess the Data**: This might include cleaning text data, handling missing values, encoding categorical variables, and normalizing numerical values.

### Step 2: Data Analysis for Insights

- **Analyse Reviews**: Analyze the review scores to understand the distribution of customer satisfaction across products.
- **Product Analysis**: Look into product categories, brands, and prices to identify popular and niche segments.

### Step 3: Recommendation System Development

Given the nature of your data, there are several approaches we can take to build the recommendation system:

1. **Content-Based Filtering**: Use product information (e.g., category, brand, tags, and description) to recommend similar items to what a user likes. This approach will require text processing and similarity measures.

2. **Collaborative Filtering**: Leverage user review scores to find similar users or products based on ratings. This method can be implemented using matrix factorization techniques like Singular Value Decomposition (SVD) or using deep learning approaches.

3. **Hybrid Method**: Combine both content-based and collaborative filtering to leverage both product attributes and user reviews for recommendations.

### Step 4: Model Implementation and Evaluation

- **Select and Implement the Model**: Depending on the chosen approach, implement the model using appropriate libraries (e.g., Scikit-learn for simpler models, TensorFlow or PyTorch for deep learning models).
- **Evaluate the Model**: Use metrics suitable for recommendation systems, such as Mean Absolute Error (MAE) for rating prediction accuracy, or precision and recall for the quality of ranked recommendations.

### Step 5: Prepare the Presentation

- Summarize the approach, methodology, and results.
- Highlight key findings from the data analysis.
- Explain the chosen recommendation system approach, its advantages, and limitations.
- Discuss the model's performance and potential areas for improvement.

Let's start with the first step, which involves loading and exploring your datasets to understand their structure and the data preprocessing needed. I'll load each dataset and provide a brief overview of their contents.
"""

import pandas as pd

# Load the datasets
user_reviews_df = pd.read_excel('User_review_data.xlsx')
product_info_df = pd.read_excel('Makeup_Products_Metadata.xlsx')

# Display the first few rows of each dataset to understand their structure
user_reviews_df.head()

product_info_df.head()

"""### User Reviews Dataset
- This dataset appears to be organized in a wide format where each row represents a user, and each column represents a product ID, with cells containing review scores. However, the preview shows that all visible values are 0, which may indicate missing or placeholder values for reviews. There are 567 columns, suggesting a wide range of products being reviewed.

### Product Information Dataset
- The `Makeup_Products_Metadata.xlsx` file contains detailed information about each product, including:
  - **Product ID**: A unique identifier for each product.
  - **Product Category**: The category to which the product belongs, such as makeup for face, lips, etc.
  - **Product Brand**: The brand of the product.
  - **Product Name**: The name of the product.
  - **Product Price [SEK]**: The price of the product in SEK (Swedish Krona).
  - **Product Description**: A brief description of the product.
  - **Product Tags**: Keywords or tags associated with the product.
  - **Product Contents**: The contents or ingredients of the product, with some fields being empty.

To carry out the data analysis for insights on customer satisfaction and product popularity, we'll proceed with the following analyses:

1. **Analyze Review Scores**:
   - Calculate average review scores for each product to understand overall customer satisfaction.
   - Assess the distribution of review scores to identify any trends or outliers.

2. **Product Analysis**:
   - Examine the distribution of products across different categories to identify which categories are most populated.
   - Analyze the distribution of products by brands to see which brands dominate the catalog.
   - Investigate the price range across different product categories to understand the pricing strategy and identify premium vs. budget segments.

Let's start with analyzing the review scores, followed by analyzing the product categories, brands, and prices. Since the review dataset appears to be in a wide format with users as rows and products as columns, I'll transform the data to calculate the average review score for each product. Then, we'll proceed with the product analysis.
"""

# Transform the user review data to a long format for easier analysis
review_data_long = user_reviews_df.melt(id_vars='User', var_name='Product ID', value_name='Review Score')

# Calculate average review score for each product
average_reviews = review_data_long.groupby('Product ID')['Review Score'].mean().reset_index()

# Analyze the distribution of review scores
review_score_distribution = average_reviews['Review Score'].describe()

# Product Category Analysis
category_distribution = product_info_df['Product Category'].value_counts()

# Brand Analysis
brand_distribution = product_info_df['Product Brand'].value_counts()

# Price Analysis
price_statistics = product_info_df['Product Price [SEK]'].describe()

review_score_distribution, category_distribution.head(), brand_distribution.head(), price_statistics

"""### Review Scores Analysis

- The average review score across products is approximately 0.049 on a scale that seems to be normalized (possibly between 0 and 1 or 0 and 5, based on the common practices for rating scales). The low mean score suggests either a low rating average or a different interpretation of the scale used.
- The standard deviation is 0.022, indicating a relatively narrow distribution of average review scores.
- The minimum and maximum average review scores are 0.0033 and 0.135, respectively, showing a wide range of customer satisfaction levels but within a low overall scoring range.
- The 25th, 50th (median), and 75th percentiles are 0.033, 0.048, and 0.063, respectively, suggesting most products receive similar scores.

### Product Category Analysis

- The most populated category is "Null" with 73 entries, indicating missing data that needs to be addressed. The next categories include "Makeup > Lips > Lipstick" with 21 products and "Makeup > Face > Blush" with 15 products, highlighting these as popular categories in the product catalog.
- Other notable categories are "Mom & Baby > Maternity Wear > Maternity Bra" and "Makeup > Lips > Liquid Lipstick," each with 14 products, indicating a diverse range of product offerings beyond just makeup.

### Brand Analysis

- The top brands by product count are "Himalaya" with 17 products, "Nykaa Cosmetics" with 14 products, "Lakme" with 13 products, "Inner Sense" with 12 products, and "Allin Exporters" with 11 products. This shows a mix of brands covering different product categories and price points.

### Price Analysis

- The average price of products is approximately 201 SEK, with a wide range of prices indicated by a standard deviation of 270.22 SEK.
- Prices range from 5 SEK to 1900 SEK, showing a broad spectrum from budget to premium products.
- The 25th, 50th (median), and 75th percentiles are 35.85 SEK, 81.1 SEK, and 250 SEK, respectively, indicating a significant number of products are priced at a more accessible level, with a smaller selection of higher-priced items.

### Insights and Implications for the Recommendation System

- The distribution of review scores suggests the need for normalization and possibly a more detailed investigation into how reviews are collected and scored.
- The diversity in product categories and brands provides an opportunity to tailor recommendations based on user preferences for specific types of products or brands.
- The wide range of prices allows for price-based filtering in recommendations, catering to users with different budget constraints.

To proceed with building the recommendation system for C's web shop, we will need to choose between content-based filtering, collaborative filtering, or a hybrid approach that combines both. Each approach has its strengths and is suited for different types of data and user needs. Here's a brief overview and how we might implement each with your datasets:

### Content-Based Filtering
- **Focus**: Recommends items similar to those a user has liked in the past, based on item features.
- **Implementation with Your Data**:
  - Use the `Product Information` dataset to extract features like product category, brand, tags, and description.
  - Process text data (tags, description) using techniques like TF-IDF or word embeddings to capture the semantic similarity between products.
  - For each user, recommend products similar to their previously rated or browsed items, based on these feature vectors.

### Collaborative Filtering
- **Focus**: Recommends items by finding similar users or items based on review scores.
- **Implementation with Your Data**:
  - Use the `User Reviews` dataset, leveraging the review scores across different products.
  - A model like matrix factorization (e.g., SVD) can be used to predict how a user might rate items they haven't reviewed yet, based on the latent user-item interaction patterns.
  - Alternatively, item-based collaborative filtering can recommend items that are similar in terms of who has reviewed them.

### Data Preprocessing Required
- **User Reviews Dataset**:
  - The dataset needs to be transformed to a long format where each row represents a user-item interaction (review), making it easier to apply collaborative filtering algorithms.
  - Address the large number of zeros (which might represent missing data) by filtering out products not reviewed by users or imputing missing values based on some strategy.
- **Product Information Dataset**:
  - Clean and preprocess text data, including product descriptions and tags.
  - Encode categorical variables like product category and brand, possibly using one-hot encoding or label encoding.
  - Handle missing values, especially in the product category and contents columns.

### Decision on Approach
Given the datasets and objectives, **a hybrid approach** might be most effective, leveraging the strengths of both content-based and collaborative filtering:
- **Content-based** aspects can capture the nuances of different products' features and user preferences for specific attributes (like brands or product types).
- **Collaborative filtering** can leverage user behavior and preferences even when explicit product features might not fully capture why a product is appealing.

### Implementation Plan
1. **Preprocess Data**: Prepare both datasets for analysis, handling missing values and encoding categorical/text features.
2. **Feature Extraction**: For content-based filtering, extract and process features from the product information dataset.
3. **Model Development**:
   - For collaborative filtering, develop a model using the review scores to predict user-item interactions.
   - For content-based filtering, calculate similarity scores between items based on their features.
4. **Recommendation Generation**: Combine the outputs of both models to generate product recommendations for users.
5. **Evaluation**: Use metrics like precision, recall, or RMSE (for rating predictions) to evaluate the effectiveness of the recommendations.
"""

# Step 1.1: Preprocess User Reviews Dataset - Transform to Long Format
# Note: A transformation has already been made in a previous step for analysis. We will formalize this for model input preparation.

# Filter out rows where all review scores are zero, assuming these represent users with no reviews.
filtered_reviews = review_data_long[review_data_long['Review Score'] > 0]

# Step 1.2: Preprocess Product Information Dataset
# Clean Text Data: For demonstration, we'll focus on handling missing values and encoding for now.

# Handle Missing Values in Product Information
product_info_clean = product_info_df.fillna({'Product Category': 'Unknown', 'Product Contents': 'Not Listed'})

# Encode Categorical Variables (simplified approach for demonstration)
# For product category and brand, we'll use a simple label encoding scheme here for demonstration purposes.
category_labels = pd.factorize(product_info_clean['Product Category'])[0]
brand_labels = pd.factorize(product_info_clean['Product Brand'])[0]

# Add encoded labels back to the dataset
product_info_clean['Category Label'] = category_labels
product_info_clean['Brand Label'] = brand_labels

# Prepare the final clean datasets
final_reviews = filtered_reviews[['Product ID', 'User', 'Review Score']]
final_product_info = product_info_clean[['Product ID', 'Product Category', 'Product Brand', 'Category Label', 'Brand Label']]

final_reviews.head(), final_product_info.head()

"""The preprocessing steps have been completed for both datasets, resulting in clean and structured data suitable for the next stages of our recommendation system development:

### User Reviews Dataset
- Transformed to a long format where each row represents a single user-product review.
- Filtered to include only rows with review scores, eliminating entries that likely represented no interaction.

### Product Information Dataset
- Missing values in `Product Category` and `Product Contents` were handled by filling them with placeholders ("Unknown" and "Not Listed", respectively).
- Categorical variables `Product Category` and `Product Brand` were encoded using a simple label encoding scheme, adding `Category Label` and `Brand Label` to the dataset for easier processing in model development.

Let's proceed with the development of the collaborative filtering model first. Collaborative filtering focuses on predicting user preferences for items based on past interactions within the user-item matrix. Given the structured user reviews dataset, we will use a matrix factorization technique, which is a common and effective method in collaborative filtering.

### Matrix Factorization Technique Overview

Matrix factorization techniques decompose the user-item interaction matrix into lower-dimensional matrices, capturing latent factors associated with users and items. Singular Value Decomposition (SVD) is one of the most popular matrix factorization techniques used for recommendation systems.

### Implementation Steps for Collaborative Filtering

1. **Prepare the User-Item Matrix**: Convert the long-format user reviews dataset into a sparse matrix where rows represent users, columns represent items (products), and cells contain review scores.

2. **Apply Matrix Factorization**: Use SVD or a similar technique to factorize the user-item matrix, identifying latent factors that represent underlying patterns in user-item interactions.

3. **Predict Missing Ratings**: Utilize the factorized matrices to predict missing ratings in the user-item matrix, enabling us to recommend items to users based on predicted scores.

4. **Generate Recommendations**: For each user, identify items with the highest predicted ratings that the user hasn't interacted with yet and recommend these items.

Given the structure of our data, we'll simplify the process for demonstration purposes, focusing on the conceptual steps. In practice, libraries such as `scikit-learn` for simpler SVD implementations or `surprise`, a Python scikit for building and analyzing recommender systems, can be used to handle these tasks more efficiently.

Let's start with preparing the user-item matrix from the user reviews dataset.
"""

from scipy.sparse import csr_matrix
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import LabelEncoder

# Encode users and products to create a sparse matrix
user_encoder = LabelEncoder()
product_encoder = LabelEncoder()

final_reviews['User ID'] = user_encoder.fit_transform(final_reviews['User'])
final_reviews['Product ID Encoded'] = product_encoder.fit_transform(final_reviews['Product ID'])

# Create a sparse matrix
num_users = final_reviews['User ID'].nunique()
num_products = final_reviews['Product ID Encoded'].nunique()

# Create the user-item matrix
user_item_matrix = csr_matrix((final_reviews['Review Score'], (final_reviews['User ID'], final_reviews['Product ID Encoded'])),
                              shape=(num_users, num_products))

# Apply Matrix Factorization - Truncated SVD
svd = TruncatedSVD(n_components=20, random_state=42)
user_item_matrix_reduced = svd.fit_transform(user_item_matrix)

# Check the shape of the reduced user-item matrix
user_item_matrix_reduced.shape

"""

The Matrix Factorization process using Truncated SVD has been successfully applied to the user-item interaction matrix, reducing it to a lower-dimensional representation with 20 latent factors. The resulting shape of the reduced matrix is (599 users, 20 latent factors), which indicates that we have successfully captured the underlying patterns in user preferences across the products reviewed.

### Next Steps for Generating Recommendations

1. **Predict Ratings**: Utilize the reduced user-item matrix to estimate ratings for all user-product pairs. This step involves reconstructing the user-item matrix from the reduced dimensions, allowing us to fill in the missing ratings.

2. **Identify Top Recommendations**: For each user, sort items based on the predicted ratings, filtering out items that the user has already reviewed. The top-ranked items per user can then be recommended.

3. **Evaluation**: To assess the effectiveness of the collaborative filtering model, we could split the original user-item interaction data into training and test sets, apply the model to the training set, and evaluate its prediction accuracy on the test set using metrics such as Root Mean Squared Error (RMSE) or Mean Absolute Error (MAE).

4. **Integrate with Content-Based Filtering**: To enhance recommendation quality, the collaborative filtering results can be combined with content-based recommendations, ensuring users receive items that are not only popular among similar users but also closely aligned with their specific interests.

"""

import numpy as np

# Step 1: Simulate Predicting Missing Ratings
# For demonstration purposes, we'll simulate the prediction process
# In practice, this would involve using the svd.inverse_transform method or similar to predict all ratings

# For demonstration, let's simulate predicted ratings for a single user
# In practice, we'd do this for all users
user_id_example = 0  # Example user ID
predicted_ratings_example = np.dot(user_item_matrix_reduced[user_id_example, :], svd.components_)

# Step 2: Identify Top Recommendations
# Filter out items the example user has already rated to focus on new recommendations
already_rated = user_item_matrix[user_id_example].nonzero()[1]

# Simulate filtering out already rated items and sorting the remaining items by predicted rating
# Here we just demonstrate the concept for the example user
predictions_filtered = [(product_encoder.inverse_transform([idx])[0], rating) for idx, rating in enumerate(predicted_ratings_example) if idx not in already_rated]

# Sort the predictions by rating in descending order to get the top recommendations
top_recommendations_example = sorted(predictions_filtered, key=lambda x: x[1], reverse=True)[:5]  # Top 5 recommendations

top_recommendations_example

"""For our example user, the collaborative filtering model has generated the following top 5 product recommendations, listed with their predicted rating values:

1. **Product ID 90254** with a predicted rating of approximately 0.536
2. **Product ID 90135** with a predicted rating of approximately 0.492
3. **Product ID 90435** with a predicted rating of approximately 0.400
4. **Product ID 90037** with a predicted rating of approximately 0.357
5. **Product ID 90545** with a predicted rating of approximately 0.338

These recommendations are based on the latent factors identified through Matrix Factorization (SVD), targeting products that this user is predicted to rate highly based on their previous ratings and the patterns learned from the entire dataset.

### Next Steps:

- **Scale Recommendation Generation**: Extend this process to generate recommendations for all users by iterating over the user base, predicting missing ratings, and identifying top recommendations for each individual.
- **Integration with Content-Based Filtering**: To enhance the recommendation quality, integrate these collaborative filtering recommendations with content-based recommendations. This hybrid approach can leverage both the similarity of items based on their content and user preferences inferred from ratings.
- **Evaluation and Tuning**: Evaluate the recommendation system using appropriate metrics (e.g., precision, recall) on a held-out test set to measure its performance and iteratively refine the model.

To proceed with the implementation of the content-based filtering component of our recommendation system, we'll focus on the product information dataset, `Makeup_Products_Metadata.xlsx`. This phase involves extracting features from product attributes such as category, brand, tags, and descriptions, and then using these features to calculate similarity scores between products. Products similar to those a user has shown interest in can then be recommended.

### Steps for Implementing Content-Based Filtering:

#### 1 Feature Extraction:
- **Text Processing**: Use techniques like TF-IDF (Term Frequency-Inverse Document Frequency) to convert product descriptions and tags into a numerical format that captures the importance of words within the documents and across the corpus of all product descriptions.
- **Categorical Data**: Encode categorical attributes like product category and brand, which can be done through one-hot encoding or similar techniques to represent these as features.

#### 2. Similarity Calculation:
- **Cosine Similarity**: Calculate the cosine similarity between products based on their feature vectors. Cosine similarity measures the cosine of the angle between two vectors, which in this context represents how similar two products are in terms of their descriptions, categories, brands, etc.

#### 3. Recommendation Generation:
- For a given user, identify products they have reviewed positively or shown interest in, and find other products that are most similar to these based on their feature vectors.
- Recommend these similar products to the user.

Let's start with feature extraction from the product information dataset, focusing initially on text data processing of product descriptions and tags, and then proceed to calculate similarity scores between products..
"""

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Step 1: Feature Extraction

# Combine product description and tags into a single text feature for each product
# For demonstration, we assume missing descriptions are replaced with an empty string
product_info_clean['Combined Text'] = product_info_clean['Product Description'].fillna('') + " " + product_info_clean['Product Tags'].fillna('')

# Use TF-IDF to convert the combined text into a matrix of TF-IDF features
tfidf_vectorizer = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf_vectorizer.fit_transform(product_info_clean['Combined Text'])

# Step 2: Calculate Cosine Similarity between products
cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

# Display the shape of the cosine similarity matrix to verify
cosine_sim.shape

"""
The feature extraction process has successfully transformed the combined product descriptions and tags into a matrix of TF-IDF features, and we've calculated the cosine similarity between all pairs of products in the dataset. The resulting cosine similarity matrix has a shape of (566, 566), indicating that we have similarity scores for each pair of products in the dataset.

### Using Cosine Similarity for Recommendation Generation

With the cosine similarity matrix in hand, we can now generate recommendations based on product similarities. For a given product, we can identify the most similar products by looking at the highest values in its corresponding row in the cosine similarity matrix.

### Example Recommendation Generation

To demonstrate how we can use this matrix to recommend products, let's select an example product and find the top 5 most similar products to it. This will illustrate how content-based recommendations can be generated for users based on their past interests."""

# Example: Generate recommendations for the first product in the dataset

# Get the product ID of the first product as an example
example_product_id = product_info_clean['Product ID'].iloc[0]

# Find the index of the example product in our dataset
example_product_idx = product_info_clean.index[product_info_clean['Product ID'] == example_product_id][0]

# Get the most similar products
sim_scores = list(enumerate(cosine_sim[example_product_idx]))

# Sort the products based on the similarity scores in descending order
sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

# Get the scores of the 5 most similar products
top_sim_scores = sim_scores[1:6]  # Skip the first one because it's the example product itself

# Get the product indices
top_product_indices = [i[0] for i in top_sim_scores]

# Get the product IDs and names
top_product_ids = product_info_clean['Product ID'].iloc[top_product_indices]
top_product_names = product_info_clean['Product Name'].iloc[top_product_indices]

# Display the recommended product IDs and names
top_product_ids, top_product_names

"""


For the example product in our dataset, the content-based filtering method has identified the following top 5 most similar products based on their descriptions and tags:

1. **Product ID 90400**: L.A. Colors Highlight & Contour Palette - Light/Medium
2. **Product ID 90141**: Makeup Revolution HD Pro Ultra Powder Contour - Fair
3. **Product ID 90276**: House Of Makeup Double Duty Kohl + Liner - Silver Lining
4. **Product ID 90233**: L.A. Girl Pro Contour Cream - Deep
5. **Product ID 90033**: Provoc Contour Correct Conceal Palette - Professional Makeup

These recommendations illustrate how content-based filtering can be used to suggest products that are similar in terms of attributes like category, brand, and product description to items a user has shown interest in. This approach can be personalized by selecting products based on a user's previous positive interactions (e.g., high review scores) and then finding and recommending similar items.

### Integrating with Collaborative Filtering for Hybrid Recommendations

The final step in building the recommendation system is to integrate these content-based recommendations with the collaborative filtering predictions made earlier. This hybrid approach can leverage both the specific attributes of products that a user may prefer and the patterns of preferences across the user base to generate a well-rounded set of recommendations.
"""