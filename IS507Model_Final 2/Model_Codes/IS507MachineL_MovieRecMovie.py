#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 12 01:28:03 2024

@author: xuchen
"""

import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import joblib
import time
import logging
import ast

class MovieRecommendationSystem:
    def __init__(self, movies_metadata_file, user_ratings_file):
        self.movies_metadata = pd.read_csv(movies_metadata_file)
        self.user_ratings = pd.read_csv(user_ratings_file)
        self.user_movie_matrix = None
        self.ml_model = None
        self.logger = self.setup_logger()
        self.preprocess_data()
        self.construct_user_movie_matrix()
        self.train_ml_model()

    def setup_logger(self):
        logging.basicConfig(
            filename="training_log.txt",
            level=logging.INFO,
            format="%(asctime)s - %(levelname)s - %(message)s"
        )
        return logging.getLogger()

    def preprocess_data(self):

        def parse_json(json_string):
            if pd.isna(json_string):
                return None
            try:
                return ast.literal_eval(json_string)
            except:
                return None

        self.movies_metadata['belongs_to_collection'] = self.movies_metadata['belongs_to_collection'].apply(parse_json)
        self.movies_metadata['genres'] = self.movies_metadata['genres'].apply(parse_json)

        self.movies_metadata['collection_id'] = self.movies_metadata['belongs_to_collection'].apply(lambda x: x['id'] if x else None)
        self.movies_metadata['genre_names'] = self.movies_metadata['genres'].apply(lambda x: [genre['name'] for genre in x] if x else [])

        self.movies_metadata['id'] = pd.to_numeric(self.movies_metadata['id'], errors='coerce')
        self.movies_metadata['vote_average'] = pd.to_numeric(self.movies_metadata['vote_average'], errors='coerce')
        self.movies_metadata['popularity'] = pd.to_numeric(self.movies_metadata['popularity'], errors='coerce')
        self.movies_metadata['title'] = self.movies_metadata['title'].astype(str)

        self.movies_metadata = self.movies_metadata.dropna(subset=['id', 'title', 'vote_average', 'popularity'])

    def construct_user_movie_matrix(self):
        self.user_movie_matrix = self.user_ratings.pivot_table(
            index='userId', columns='movieId', values='rating'
        ).fillna(0)

    def train_ml_model(self):
        start_time = time.time()
    
        X = []
        y = []
    
        for user_id, user_ratings in self.user_movie_matrix.iterrows():
            for movie_id, rating in user_ratings.items():
                if rating > 0:
                    movie_features = self.get_movie_features(movie_id)
                    if movie_features and not any([feature is None or pd.isna(feature) for feature in movie_features]):
                        X.append(movie_features)
                        y.append(rating)
    
        X = np.array(X)
        y = np.array(y)
    
        if len(X) == 0 or len(y) == 0:
            raise ValueError("Training data is empty. Please check the user and movie datasets.")
    
        self.ml_model = Ridge(alpha=1.0)
        self.ml_model.fit(X, y)
    
        elapsed_time = time.time() - start_time
        self.logger.info(f"Model trained in {elapsed_time:.2f} seconds with {len(X)} samples.")
        joblib.dump(self.ml_model, 'ridge_model.pkl')
    
        predictions = self.ml_model.predict(X)
        rmse = np.sqrt(mean_squared_error(y, predictions))
        self.logger.info(f"Training RMSE: {rmse:.2f}")

        
    def get_movie_features(self, movie_id):
        movie = self.movies_metadata[self.movies_metadata['id'] == movie_id]
        if not movie.empty:
            movie = movie.iloc[0]
            return [
                movie['popularity'] if pd.notna(movie['popularity']) else 0,
                movie['vote_average'] if pd.notna(movie['vote_average']) else 0,
                len(movie['genre_names']) if isinstance(movie['genre_names'], list) else 0,
                movie['collection_id'] if pd.notna(movie['collection_id']) else 0
            ]
        return None

    def predict_user_rating(self, user_id, movie_id):
        movie_features = self.get_movie_features(movie_id)
        if movie_features is None:
            return 0  # Return a default value if movie features are not available
        
        # Use the trained ML model to predict the rating
        predicted_rating = self.ml_model.predict([movie_features])[0]
        return predicted_rating

    def recommend_movies(self, movie_id, user_id, n_recommendations=5):
        if movie_id not in self.movies_metadata['id'].values:
            raise ValueError(f"Movie ID {movie_id} not found in the dataset.")
    
        # Get features of the target movie
        popularity, vote_average, genre_count, target_collection_id = self.get_movie_features(movie_id)
        target_genres = set(self.movies_metadata[self.movies_metadata['id'] == movie_id]['genre_names'].iloc[0])
    
        # Step 1: Recommend movies from the same collection
        collection_recommendations = self.movies_metadata[
            (self.movies_metadata['collection_id'] == target_collection_id) &
            (self.movies_metadata['id'] != movie_id)
        ]
    
        # Step 2: Recommend movies with the same genres
        exact_genre_recommendations = self.movies_metadata[
            (self.movies_metadata['genre_names'].apply(lambda x: set(x) == target_genres)) &
            (self.movies_metadata['id'] != movie_id) &
            (~self.movies_metadata['id'].isin(collection_recommendations['id']))
        ]
        
        if len(exact_genre_recommendations) < n_recommendations:
            partial_genre_recommendations = self.movies_metadata[
                (self.movies_metadata['genre_names'].apply(lambda x: bool(set(x) & target_genres))) &
                (self.movies_metadata['id'] != movie_id) &
                (~self.movies_metadata['id'].isin(collection_recommendations['id']))
            ]
        else:
            partial_genre_recommendations = pd.DataFrame()
    
        genre_recommendations = pd.concat([exact_genre_recommendations, partial_genre_recommendations])
        genre_recommendations = genre_recommendations.sort_values(by='popularity', ascending=False).head(n_recommendations)
    
        remaining_movies = self.movies_metadata[
            (~self.movies_metadata['id'].isin(collection_recommendations['id'])) &
            (~self.movies_metadata['id'].isin(genre_recommendations['id'])) &
            (self.movies_metadata['id'] != movie_id)
        ]
    
        all_recommendations = pd.concat([
            collection_recommendations.assign(rec_type='collection'),
            genre_recommendations.assign(rec_type='genre'),
            remaining_movies.assign(rec_type='popularity')
        ], ignore_index=True)[['id', 'title', 'rec_type', 'popularity', 'vote_average']]

    
        # Add predicted_rating column
        all_recommendations['predicted_rating'] = all_recommendations['id'].apply(
            lambda movie_id: self.predict_user_rating(user_id, movie_id)
        )
    
        # Sort recommendations by score and predicted_rating
        all_recommendations['score'] = all_recommendations.apply(
            lambda row: 3 if row['rec_type'] == 'collection' else (2 if row['rec_type'] == 'genre' else 1),
            axis=1
        )
        final_recommendations = all_recommendations.sort_values(
            by=['score', 'predicted_rating', 'vote_average', 'popularity'],
            ascending=[False, False, False, False]
        ).head(n_recommendations)
    
        return final_recommendations[['id', 'title', 'vote_average', 'popularity', 'rec_type', 'predicted_rating']]

    
    
               
        print("All Recommendations:")
        print(all_recommendations[['id', 'title', 'rec_type', 'predicted_rating']].head(10))

    
    
        all_recommendations['predicted_rating'] = all_recommendations['id'].apply(
            lambda movie_id: self.predict_user_rating(user_id, movie_id)
        )
    
        all_recommendations['score'] = all_recommendations.apply(
            lambda row: 3 if row['rec_type'] == 'collection' else (2 if row['rec_type'] == 'genre' else 1),
            axis=1
        )
    
        final_recommendations = all_recommendations.sort_values(
            by=['score', 'predicted_rating', 'vote_average', 'popularity'],
            ascending=[False, False, False, False]
        ).head(n_recommendations)
    
        return final_recommendations[['id', 'title', 'vote_average', 'popularity', 'rec_type', 'predicted_rating']]
    
    
    

# Example usage
if __name__ == "__main__":
    recommender = MovieRecommendationSystem(
        '/Users/xuchen/Desktop/IS507Model_Final/DataSets/Movies.csv',
        '/Users/xuchen/Desktop/IS507Model_Final/DataSets/ratings_small.csv'
    )


    search_title = 'Ant-Man'.lower()    #Movie name must be corrected,fuzzy search won't work

    Likedmovie = recommender.movies_metadata[recommender.movies_metadata['title'].str.lower() == search_title]['id'].iloc[0]

    recommendations = recommender.recommend_movies(Likedmovie, user_id=1)
    print("Recommendations:")
    print(recommendations)

    # Load and test the saved model
    loaded_model = joblib.load('ridge_model.pkl')
    print("Loaded model successfully.")

