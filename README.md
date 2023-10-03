# Movie-recommendation
#Movie recommendation based on content_based_filtering

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

def get_movie_recommendations(movie_title):

    # Load the movie dataset
    movies = pd.read_csv('File location')

    # Create a TF-IDF matrix
    tfidf = TfidfVectorizer()
    tfidf_matrix = tfidf.fit_transform(movies['description'])

    # Calculate the cosine similarity between movies
    cosine_similarities = cosine_similarity(tfidf_matrix, tfidf_matrix)

    # Get the index of the input movie
    movie_index = movies[movies['title'] == movie_title].index

    # If the movie is not found in the dataset, return None
    if len(movie_index) == 0:
        return None

    # Get the cosine similarity scores for the input movie with all movies
    similarity_scores = cosine_similarities[movie_index[0]]

    # Sort the similarity scores in descending order
    sorted_similarity_scores = sorted(enumerate(similarity_scores), key=lambda x: x[1], reverse=True)

    # Get the top 10 most similar movies
    recommended_movies = []
    for i in range(1, 11):
        recommended_movie_index = sorted_similarity_scores[i][0]
        recommended_movie_title = movies.iloc[recommended_movie_index]['title']
        recommended_movie_type = movies.iloc[recommended_movie_index]['type']
        recommended_movie_release_year= movies.iloc[recommended_movie_index]['release_year']

        
        recommended_movie = {
            'title': recommended_movie_title,
            'type': recommended_movie_type,
            'release_year': recommended_movie_release_year
        }

        recommended_movies.append(recommended_movie)
    return recommended_movies

movie_title = 'The Starling'
recommended_movies = get_movie_recommendations(movie_title)

print('Recommended movies for "{}":'.format(movie_title))
for movie in recommended_movies:
    print('Title: {}, Type: {}, Release_year: {}'.format(movie['title'], movie['type'], movie['release_year']))
