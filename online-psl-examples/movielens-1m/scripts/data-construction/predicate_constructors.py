"""
Define methods for constructing and writing PSL predicates from raw movielens data.
"""

import numpy as np
import os
import pandas as pd

from sklearn.naive_bayes import MultinomialNB
from surprise.prediction_algorithms.matrix_factorization import NMF
from surprise.reader import Reader
from surprise.dataset import Dataset


def write(data, predicate_name, path):
    """
    Write a series to disk representing a PSL predicate.
    :param data: The pandas Series or DataFrame being written to disk.
    :param predicate_name: The predicate the data represents
    :param path: The path for this data.
    :return:
    """
    # path to this file relative to caller
    data.to_csv(os.path.join(path, predicate_name + '.txt'), sep='\t', header=False, index=True)


def top_n_cosine_sim(matrix, index, n):
    row_norms = [np.linalg.norm(row) for row in matrix]
    normalized_matrix = np.array([matrix[i] / row_norms[i] for i in range(len(matrix))])
    similarity_frame_data = np.matmul(normalized_matrix, normalized_matrix.T)
    similarity_df = pd.DataFrame(data=similarity_frame_data,
                                 index=index, columns=index)

    # take top n for each movie to define pairwise blocks
    top_n_similarity_frame = pd.DataFrame(index=index, columns=range(n))
    for m in index:
        top_n_similarity_frame.loc[m, :] = similarity_df.loc[m].nlargest(n).index

    flattened_frame = top_n_similarity_frame.values.flatten()
    sim_index = np.array([[i] * 50 for i in index]).flatten()
    sim_index = pd.MultiIndex.from_arrays([sim_index, flattened_frame])
    similarity_series = pd.Series(data=1, index=sim_index)

    # populate the item_content_similarity_series with the similarity value
    for i in sim_index:
        similarity_series.loc[i] = np.clip(similarity_df.loc[i[0], i[1]], 0.0, 1.0)

    return similarity_series


def average_item_rating_predicate(observed_ratings_df, path, unique_movies, fill_na=True):
    """
    Average item rating predicates.
    """
    avg_rating_series = observed_ratings_df.loc[:, 'rating'].reset_index()[["movieId", "rating"]].groupby("movieId").mean()
    if fill_na:
        avg_rating_series = avg_rating_series.reindex(unique_movies, fill_value=avg_rating_series.mean()['rating'])
    write(avg_rating_series, 'avg_item_rating_obs', path)


def average_user_rating_predicate(observed_ratings_df, path, unique_users, fill_na=True):
    """
    Average user rating predicates.
    """
    avg_rating_series = observed_ratings_df.loc[:, 'rating'].reset_index()[["userId", "rating"]].groupby("userId").mean()
    if fill_na:
        avg_rating_series = avg_rating_series.reindex(unique_users, fill_value=avg_rating_series.mean()['rating'])
    write(avg_rating_series, 'avg_user_rating_obs', path)


def item_predicate(observed_ratings_df, truth_ratings_df, path):
    """
    Item scoping predicates
    """
    observed_ratings_series = observed_ratings_df.loc[:, 'rating']
    truth_ratings_series = truth_ratings_df.loc[:, 'rating']

    # obs
    item_list = pd.concat([observed_ratings_series, truth_ratings_series], join='outer').reset_index()['movieId'].unique()
    item_series = pd.Series(data=1, index=item_list)
    write(item_series, 'item_obs', path)


def nb_ratings_predicate(observed_ratings_df, truth_ratings_df, user_df, movies_df, path):
    """
    nb_ratings Predicates. The multinomial naive bayes multi-class classifier predictions
    """

    # Build user-movie rating vector frame.
    dummified_user_df = pd.get_dummies(user_df.drop('zip', axis=1).astype({'age': object, 'occupation': object}))
    train_user_movie_rating_vector_df = (
        observed_ratings_df.drop('timestamp', axis=1).join(dummified_user_df, on='userId').join(movies_df.drop('movie title', axis=1), on='movieId'))
    test_user_movie_rating_vector_df = (
        truth_ratings_df.drop('timestamp', axis=1).join(dummified_user_df, on='userId').join(movies_df.drop('movie title', axis=1), on='movieId'))

    # Fit naive bayes model with Laplace smoothing parameter alpha=1.0.
    nb_model = MultinomialNB(alpha=1.0)
    nb_model.fit(train_user_movie_rating_vector_df.drop('rating', axis=1),
                 train_user_movie_rating_vector_df.rating.astype(str))

    # Make predictions for the user item pairs in the truth frame
    predictions = pd.DataFrame(nb_model.predict(test_user_movie_rating_vector_df.drop('rating', axis=1)),
                               index=test_user_movie_rating_vector_df.index)

    write(predictions, 'nb_rating_obs', path)


def nmf_ratings_predicate(observed_ratings_df, truth_ratings_df, path):
    """
    Build the nmf_ratings predicates.
    The non-negative matrix factorization implementation from the surprise python library performs SGD optimization
    without using only the observed ratings thus the matrix does not need to be filled.
    """
    nmf_model = NMF()
    # Specifies the lower and upper bounds of the ratings.
    reader = Reader(rating_scale=(0.2, 1))
    train_dataset = Dataset.load_from_df(df=observed_ratings_df.reset_index().loc[:, ['userId', 'movieId', 'rating']],
                                         reader=reader)
    nmf_model.fit(train_dataset.build_full_trainset())

    # make predictions
    predictions = pd.DataFrame(index=truth_ratings_df.index, columns=['rating'])

    for row in truth_ratings_df.loc[:, ['rating']].iterrows():
        uid = row[0][0]
        iid = row[0][1]
        predictions.loc[(uid, iid), 'rating'] = np.clip(nmf_model.predict(uid, iid).est, 0.0, 1.0)

    write(predictions, 'nmf_rating_obs', path)


def rated_predicate(observed_ratings_df, truth_ratings_df, path, partition):
    """
    Rated Predicates.
    These predicates are {0,1} valued predicates that are used as blocks so only those predicates
    that will be evaluated on or were observed are used in ground terms.
    """
    observed_ratings_series = observed_ratings_df.loc[:, 'rating']
    truth_ratings_series = truth_ratings_df.loc[:, 'rating']

    # obs
    rated_series = pd.concat([observed_ratings_series, truth_ratings_series], join='outer')
    rated_series.loc[:, :] = 1
    write(rated_series, 'rated_' + partition, path)


def target_predicate(truth_ratings_df, path, partition):
    """
    Target Predicates.
    These predicates are {0,1} valued predicates that are used as blocks so only those predicates
    that will be evaluated on or were observed are used in ground terms.
    """
    # truth
    target_dataframe = truth_ratings_df.loc[:, []]
    target_dataframe['value'] = 1
    write(target_dataframe, 'target_' + partition, path)


def user_predicate(observed_ratings_df, truth_ratings_df, path):
    """
    User Predicates.
    These predicates are {0,1} valued predicates that are used as blocks so only those predicates
    that will be evaluated on are inferred.
    """
    observed_ratings_series = observed_ratings_df.loc[:, 'rating']

    truth_ratings_series = truth_ratings_df.loc[:, 'rating']
    # obs
    user_list = pd.concat([observed_ratings_series, truth_ratings_series], join='outer').reset_index()['userId'].unique()
    user_series = pd.Series(data=1, index=user_list)
    write(user_series, 'user_obs', path)


def ratings_predicate(ratings_df, path, partition, write_value=True):
    """
    Ratings Predicates.
    The open predicates that will be inferred in the movielens dataset.
    """
    ratings_frame = ratings_df.loc[:, ['rating']]

    if write_value:
        write(ratings_frame, 'rating_' + partition, path)
    else:
        write(ratings_frame.loc[:, []], 'rating_' + partition, path)


def sim_content_predicate(movies_df, path):
    """
    Similar item content predicates.
    These predicates represent the cosine similarity between two items based on movie genres and take on
    values in the range (0, 1).
    """
    movie_genres_df = movies_df.drop('movie title', axis=1)

    # Cosine similarity
    movie_genres_matrix = movie_genres_df.values
    write(top_n_cosine_sim(movie_genres_matrix, movie_genres_df.index, 50), 'sim_content_items_obs', path)


def sim_items_predicate(observed_ratings_df, path):
    """
    Item Similarity Predicate: sim_cosine_items, built only from observed ratings.
    """
    # Cosine similarity.
    movie_ratings_matrix = observed_ratings_df.rating.unstack().fillna(0).T.values
    write(top_n_cosine_sim(movie_ratings_matrix, observed_ratings_df.rating.unstack().columns, 50), 'sim_items_obs', path)


def sim_users_predicate(observed_ratings_df, path):
    """
    User Similarity Predicate: sim_cosine_users, built only from observed ratings
    """
    # Cosine similarity.
    user_ratings_matrix = observed_ratings_df.rating.unstack().fillna(0).values
    write(top_n_cosine_sim(user_ratings_matrix, observed_ratings_df.rating.unstack().index, 50), 'sim_users_obs', path)
