6#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 13 17:14:39 2024

@author: xuchen
"""
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 13 17:14:39 2024

@author: xuchen
"""
from IS507MachineL_MovieRecMovie import MovieRecommendationSystem as BaseRecommendationSystem
from UserBasedREC_KNN_Ridge import MovieRecommendationSystem as PredictionSystem
import pandas as pd
import numpy as np

class IntegratedRecommendationSystem(BaseRecommendationSystem, PredictionSystem):
    def __init__(self, movies_metadata_file, user_ratings_file):
        BaseRecommendationSystem.__init__(self, movies_metadata_file, user_ratings_file)
        PredictionSystem.__init__(self, movies_metadata_file, user_ratings_file)

    def predict_user_rating(self, user_id, movie_id):
        user_ratings = self.user_ratings[self.user_ratings['userId'] == user_id]
        user_mean_rating = user_ratings['rating'].mean()

        movie_ratings = self.user_ratings[self.user_ratings['movieId'] == movie_id]
        movie_mean_rating = movie_ratings['rating'].mean()

        if np.isnan(user_mean_rating):
            user_mean_rating = self.user_ratings['rating'].mean()
        if np.isnan(movie_mean_rating):
            movie_mean_rating = self.user_ratings['rating'].mean()

        predicted_rating = (user_mean_rating + movie_mean_rating) / 2

        predicted_rating += np.random.normal(0, 0.5)

        return max(1, min(5, predicted_rating))

    def recommend_movies_for_user(self, user_id, n_recommendations=5):
        print(f"Generating recommendations for user {user_id}")
        
        # Get movies not rated by the user
        user_ratings = self.user_ratings[self.user_ratings['userId'] == user_id]
        print(f"Number of ratings by user {user_id}: {len(user_ratings)}")
        
        rated_movies = user_ratings['movieId']
        unrated_movies = self.movies_metadata[~self.movies_metadata['id'].isin(rated_movies)]
    
        if unrated_movies.empty:
            print(f"User {user_id} has no unrated movies.")
            return pd.DataFrame()
    
        # Predict ratings for each unrated movie
        def safe_predict(movie_id):
            try:
                rating = self.predict_user_rating(user_id, movie_id)
                print(f"Predicted rating by user {user_id} for movie {movie_id}: {rating}")
                return rating
            except Exception as e:
                print(f"Error predicting rating for movie {movie_id}: {e}")
                return np.nan
    
        unrated_movies['predicted_rating'] = unrated_movies['id'].apply(safe_predict)
    
        # Remove invalid predictions
        unrated_movies = unrated_movies.dropna(subset=['predicted_rating'])
    
        if unrated_movies.empty:
            print(f"All predicted ratings for user {user_id}'s unrated movies are invalid.")
            return pd.DataFrame()
    
        # Sort by predicted rating
        recommendations = unrated_movies.sort_values(by='predicted_rating', ascending=False).head(n_recommendations)
    
        return recommendations[['id', 'title', 'predicted_rating', 'popularity', 'vote_average']]




# Main program call
if __name__ == "__main__":
    recommender = IntegratedRecommendationSystem(
        '/Users/xuchen/Desktop/IS507Model_Final/DataSets/Movies.csv',
        '/Users/xuchen/Desktop/IS507Model_Final/DataSets/ratings_small.csv'
    )

    while True:
        user_input = input("Please enter the target user ID (enter 'q' to exit the program): ")
        if user_input.lower() == 'q':
            print("Program exited.")
            break

        try:
            user_id = int(user_input)
            recommendations = recommender.recommend_movies_for_user(user_id)
            if recommendations.empty:
                print(f"No movies to recommend for user {user_id}.")
            else:
                print(f"Recommended movies for user {user_id}:")
                print(recommendations)
        except ValueError:
            print("Please enter a valid user ID (integer).")
        except Exception as e:
            print(f"An error occurred: {e}")

