#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 13 15:39:04 2024

@author: xuchen
"""

import pandas as pd
import numpy as np
from sklearn.neighbors import NearestNeighbors, KNeighborsRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, mean_absolute_error
import ast

class MovieRecommendationSystem:
    def __init__(self, movies_metadata_file, user_ratings_file):
        self.movies_metadata = pd.read_csv(movies_metadata_file)
        self.user_ratings = pd.read_csv(user_ratings_file)
        self.user_movie_matrix = None
        self.ridge_model = None
        self.knn_model = None
        self.preprocess_data()
        self.construct_user_movie_matrix()
        self.train_models()

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
        self.movies_metadata = self.movies_metadata.dropna(subset=['id', 'title', 'vote_average', 'popularity'])

    def construct_user_movie_matrix(self):
        self.user_movie_matrix = self.user_ratings.pivot_table(index='userId', columns='movieId', values='rating').fillna(0)

    def train_models(self):
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

        self.ridge_model = Ridge(alpha=1.0)
        self.ridge_model.fit(X, y)

        self.knn_model = KNeighborsRegressor(n_neighbors=5)
        self.knn_model.fit(X, y)

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
            return 0, 0
        ridge_prediction = self.ridge_model.predict([movie_features])[0]
        knn_prediction = self.knn_model.predict([movie_features])[0]
        return ridge_prediction, knn_prediction

    def recommend_movies(self, movie_id, user_id, n_recommendations=5):
         if movie_id not in self.movies_metadata['id'].values:
             raise ValueError(f"Movie ID {movie_id} not found in the dataset.")

         popularity, vote_average, genre_count, target_collection_id = self.get_movie_features(movie_id)
         target_genres = set(self.movies_metadata[self.movies_metadata['id'] == movie_id]['genre_names'].iloc[0])

         collection_recommendations = self.movies_metadata[
             (self.movies_metadata['collection_id'] == target_collection_id) &
             (self.movies_metadata['id'] != movie_id)
         ].sort_values(by='popularity', ascending=False).head(n_recommendations)

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
         ])

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

    def evaluate_prediction_accuracy(self, user_id):
        user_ratings = self.user_ratings[self.user_ratings['userId'] == user_id]
        
        # Obtain control group
        control_group = user_ratings.sample(n=5)
        
        # get data that excluding control group
        training_data = user_ratings[~user_ratings.index.isin(control_group.index)]
        
        # train with training data
        X_train = []
        y_train = []
        for _, row in training_data.iterrows():
            movie_features = self.get_movie_features(row['movieId'])
            if movie_features:
                X_train.append(movie_features)
                y_train.append(row['rating'])
        
        if len(X_train) > 0:
            self.ridge_model.fit(X_train, y_train)
            self.knn_model.fit(X_train, y_train)
        
        # check accuracy with control group
        results = []
        for _, row in control_group.iterrows():
            movie_id = row['movieId']
            actual_rating = row['rating']
            ridge_prediction, knn_prediction = self.predict_user_rating(user_id, movie_id)
            results.append({
                'userId': user_id,
                'movieId': movie_id,
                'actual_rating': actual_rating,
                'ridge_prediction': ridge_prediction,
                'knn_prediction': knn_prediction,
                'timestamp': row['timestamp']
            })
        
        results_df = pd.DataFrame(results)
        
        ridge_rmse = np.sqrt(mean_squared_error(results_df['actual_rating'], results_df['ridge_prediction']))
        ridge_mae = mean_absolute_error(results_df['actual_rating'], results_df['ridge_prediction'])
        
        knn_rmse = np.sqrt(mean_squared_error(results_df['actual_rating'], results_df['knn_prediction']))
        knn_mae = mean_absolute_error(results_df['actual_rating'], results_df['knn_prediction'])
        
        best_model = 'Ridge' if ridge_rmse < knn_rmse else 'KNN'
        
        return {
            'Ridge_RMSE': ridge_rmse,
            'Ridge_MAE': ridge_mae,
            'KNN_RMSE': knn_rmse,
            'KNN_MAE': knn_mae,
            'Best_Model': best_model,
            'n_test': len(results_df),
            'control_group': control_group,
            'results': pd.DataFrame(results)
        }

# main function
if __name__ == "__main__":
    recommender = MovieRecommendationSystem(
        '/Users/xuchen/Desktop/IS507Model_Final/DataSets/Movies.csv',
        '/Users/xuchen/Desktop/IS507Model_Final/DataSets/ratings_small.csv'
    )

    while True:
        user_input = input("Please enter target user ID (input 'q' to quit): ")
        if user_input.lower() == 'q':
            print("Quit.")
            break

        try:
            target_user_id = int(user_input)
        except ValueError:
            print("Please enter a valid user ID!")
            continue

        try:
            # Prediction accuracy test
            accuracy_results = recommender.evaluate_prediction_accuracy(target_user_id)

            print(f"User {target_user_id} control group:")
            print(accuracy_results['control_group'])

            print(f"\nEvaluation results for user {target_user_id}:")
            print(f"Ridge RMSE: {accuracy_results['Ridge_RMSE']:.4f}")
            print(f"Ridge MAE: {accuracy_results['Ridge_MAE']:.4f}")
            print(f"KNN RMSE: {accuracy_results['KNN_RMSE']:.4f}")
            print(f"KNN MAE: {accuracy_results['KNN_MAE']:.4f}")
            print(f"Best Model: {accuracy_results['Best_Model']}")
            print(f"Number of test movies: {accuracy_results['n_test']}")

            print("\nEvaluation details (including actual ratings and predicted ratings):")
            # Dynamically select the best model's predictions
            results = accuracy_results['results']
            if accuracy_results['Best_Model'] == 'Ridge':
                results['best_prediction'] = results['ridge_prediction']
            else:
                results['best_prediction'] = results['knn_prediction']

            # Print final results
            print(results[['movieId', 'actual_rating', 'best_prediction']])

            # Current user's rating data
            current_user_ratings = recommender.user_ratings[
                recommender.user_ratings['userId'] == target_user_id
            ]
            
            # Extract control group
            control_group = current_user_ratings.sample(n=5, random_state=42)
            training_data = current_user_ratings[
                ~current_user_ratings.index.isin(control_group.index)
            ]
            
            print(f"Number of movies used for training user {target_user_id}: {len(training_data)}")

        except KeyError:
            print(f"User ID {target_user_id} does not exist. Please enter a valid user ID!")
        except Exception as e:
            print(f"An error occurred: {e}")
