"""
Construct PSL formatted data from raw movielens data.
"""

import pandas as pd
import numpy as np
import os
import shutil

import predicate_constructors

DIRNAME = os.path.dirname(__file__)
BASE_DATA_PATH = os.path.join(DIRNAME, "../../data")
RAW_RATINGS_PATH = os.path.join(BASE_DATA_PATH, "ml-1m/ratings.dat")
RAW_MOVIES_PATH = os.path.join(BASE_DATA_PATH, "ml-1m/movies.dat")
N_FOLDS = 10
SAMPLE_PROPORTION = 0.7  # The sample proportion for each fold.
INITIAL_PROPORTION = 1 / 3  # The proportion of the fold data that is initially observed.
NUM_PARTITIONS = 20  # The number of partitions of the initial truth dataset.
TIMESERIES_PROPORTION_OBS = 0.2

# Command constants.
WRITE_INFERRED_COMMAND = "WRITEINFERREDPREDICATES"
ADD = 'ADDATOM'
OBSERVE = 'OBSERVEATOM'
UPDATE = 'UPDATEATOM'
DELETE = 'DELETEATOM'

# Partition names
OBS = 'obs'
TARGET = 'target'
TRUTH = 'truth'


def construct_predicates():
    """
    Create data directory to write output to.
    """
    if not os.path.exists(BASE_DATA_PATH):
        os.makedirs(BASE_DATA_PATH)

    """
    Assuming that the raw data already exists in the data directory.
    """
    movies_df, ratings_df = load_dataframes()
    movies_df, ratings_df = filter_dataframes(movies_df, ratings_df)
    ratings_df_list = sample_randomly(ratings_df)

    for fold, fold_ratings_df in enumerate(ratings_df_list):
        print("Constructing fold #" + str(fold).zfill(2))

        time_series_out_directory = os.path.join(BASE_DATA_PATH, 'movielens-1m/movielens-1m_time_series/' + str(fold).zfill(2) + '/eval/')
        print("Making " + time_series_out_directory)
        if not os.path.exists(time_series_out_directory):
            os.makedirs(time_series_out_directory)

        online_out_directory = os.path.join(BASE_DATA_PATH, 'movielens-1m/movielens-1m_online/' + str(fold).zfill(2) + '/eval/')
        print("Making " + online_out_directory)
        if not os.path.exists(online_out_directory):
            os.makedirs(online_out_directory)

        # Sort ratings frame by timestep.
        fold_ratings_df = fold_ratings_df.sort_values(by='timestamp')

        # Grab movies in movie df for fold.
        fold_movies_df = movies_df.loc[fold_ratings_df.index.get_level_values('movieId').unique()]

        # Grab unique users and movies in ratings df for fold.
        fold_unique_users = fold_ratings_df.index.get_level_values('userId').unique()
        fold_unique_movies = fold_ratings_df.index.get_level_values('movieId').unique()

        # Get the observations/truth split for this fold.
        observed_ratings_df = fold_ratings_df.iloc[: int(fold_ratings_df.shape[0] * INITIAL_PROPORTION)]
        truth_ratings_df = fold_ratings_df.iloc[int(fold_ratings_df.shape[0] * INITIAL_PROPORTION):]
        partitioned_truth_ratings = partition(truth_ratings_df)

        # Construct predicates that are static for this fold.
        # print("Constructing static predicates.")
        construct_static_predicates(observed_ratings_df, truth_ratings_df, fold_movies_df,
                                    time_series_out_directory, online_out_directory)

        # Construct predicates that are dynamic with time.
        print("Constructing dynamic predicates.")
        construct_dynamic_predicates(observed_ratings_df, partitioned_truth_ratings,
                                     fold_unique_users, fold_unique_movies,
                                     time_series_out_directory, online_out_directory)


def construct_client_commands(prev_observed_ratings, observed_ratings, prev_truth_ratings, current_truth_ratings, time_step):
    add_targets_command_list = []
    update_target_command_list = []
    add_observation_command_list = []
    observe_command_list = []
    update_observation_command_list = []
    extra_commands = [WRITE_INFERRED_COMMAND + "\t'./inferred-predicates/{:02d}'".format(time_step)]

    if time_step > 0:
        # Observe and add ratings atoms.
        new_targets_df = current_truth_ratings.loc[current_truth_ratings.index.difference(prev_truth_ratings.index)].reset_index()
        add_targets_command_list += df_to_command(new_targets_df.loc[:, ['userId', 'movieId']],
                                                  new_targets_df.loc[:, []],
                                                  ADD, TARGET, 'rating')

        observed_ratings_df = observed_ratings.loc[observed_ratings.index.difference(prev_observed_ratings.index)].reset_index()
        observe_command_list += df_to_command(
            observed_ratings_df.loc[:, ['userId', 'movieId']],
            observed_ratings_df.loc[:, ['rating']],
            OBSERVE, OBS, 'rating')

        # Add rated atoms (Assumed that new rated predicates are introduced only through truths).
        new_rated_df = current_truth_ratings.loc[current_truth_ratings.index.difference(prev_truth_ratings.index)].reset_index()
        add_observation_command_list += df_to_command(
            new_rated_df.loc[:, ['userId', 'movieId']],
            new_rated_df.loc[:, ['rating']].clip(1, 1),
            ADD, OBS, 'rated')

        # Add and update target atoms.
        add_observation_command_list += df_to_command(
            new_rated_df.loc[:, ['userId', 'movieId']],
            new_rated_df.loc[:, ['rating']].clip(1, 1),
            ADD, OBS, 'target')

        observed_targets = prev_truth_ratings.loc[prev_truth_ratings.index.difference(current_truth_ratings.index)].reset_index()
        update_target_command_list += df_to_command(
            observed_targets.loc[:, ['userId', 'movieId']],
            observed_targets.loc[:, ['rating']].clip(0, 0),
            UPDATE, OBS, 'target')

        # Update averages.
        seen_user_avg = observed_ratings['rating'].reset_index()[["userId", "rating"]].groupby("userId").mean().reset_index()
        update_observation_command_list += df_to_command(
            seen_user_avg.loc[:, ['userId']],
            seen_user_avg.loc[:, ['rating']],
            UPDATE, OBS, 'avg_user_rating'
        )

        seen_movie_avg = observed_ratings.swaplevel()['rating'].reset_index()[["movieId", "rating"]].groupby("movieId").mean().reset_index()
        update_observation_command_list += df_to_command(
            seen_movie_avg.loc[:, ['movieId']],
            seen_movie_avg.loc[:, ['rating']],
            UPDATE, OBS, 'avg_item_rating'
        )

    command_list = (add_targets_command_list + update_target_command_list + observe_command_list
                    + add_observation_command_list + update_observation_command_list
                    + extra_commands)

    return command_list


def construct_dynamic_predicates(observed_ratings_df, partitioned_truth_ratings, fold_unique_users, fold_unique_movies,
                                 time_series_out_directory, online_out_directory):
    # Start initial observations.
    # These dataframes will grow with each timestep by adding some or all of the previous timestep's target set.
    online_aggregated_observed_ratings = observed_ratings_df
    time_series_aggregated_observed_ratings = observed_ratings_df
    time_series_aggregated_truth_ratings = partitioned_truth_ratings[0]

    # Initialize empty list of commands.
    time_series_command_list = []
    online_command_list = []

    # Dynamic movielens predicates.
    for time_step in np.arange(len(partitioned_truth_ratings)):
        print("Constructing predicates for time step: " + str(time_step).zfill(2))

        # Set the shared path between these predicates.
        time_series_path = os.path.join(time_series_out_directory, str(time_step).zfill(2))
        if not os.path.exists(time_series_path):
            os.makedirs(time_series_path)

        online_path = os.path.join(online_out_directory, str(time_step).zfill(2))
        if not os.path.exists(online_path):
            os.makedirs(online_path)

        # Update rating observations.
        online_prev_observed_ratings = online_aggregated_observed_ratings.copy()
        time_series_prev_observed_ratings = time_series_aggregated_observed_ratings.copy()
        time_series_prev_truth_ratings = time_series_aggregated_truth_ratings.copy()
        if time_step > 0:
            online_aggregated_observed_ratings = online_aggregated_observed_ratings.append(partitioned_truth_ratings[time_step - 1],
                                                                                           ignore_index=False)
            time_series_aggregated_truth_ratings = time_series_aggregated_truth_ratings.append(partitioned_truth_ratings[time_step],
                                                                                               ignore_index=False)

            time_series_observations = partitioned_truth_ratings[time_step - 1].sample(frac=TIMESERIES_PROPORTION_OBS)
            time_series_aggregated_observed_ratings = time_series_aggregated_observed_ratings.append(time_series_observations,
                                                                                                     ignore_index=False)
            time_series_aggregated_truth_ratings = time_series_aggregated_truth_ratings.drop(time_series_observations.index)

        # Get client commands for this timestep.
        online_command_list += construct_client_commands(online_prev_observed_ratings, online_aggregated_observed_ratings,
                                                         pd.concat(partitioned_truth_ratings[time_step-1:]),
                                                         pd.concat(partitioned_truth_ratings[time_step:]),
                                                         time_step)
        time_series_command_list += construct_client_commands(time_series_prev_observed_ratings, time_series_aggregated_observed_ratings,
                                                              time_series_prev_truth_ratings, time_series_aggregated_truth_ratings,
                                                              time_step)

        # Construct and write the predicates for timestamp.
        # Rating predicate.
        predicate_constructors.ratings_predicate(time_series_aggregated_observed_ratings, time_series_path, OBS)
        predicate_constructors.ratings_predicate(time_series_aggregated_truth_ratings,
                                                 time_series_path, TARGET, write_value=False)
        predicate_constructors.ratings_predicate(time_series_aggregated_truth_ratings,
                                                 time_series_path, TRUTH)

        predicate_constructors.ratings_predicate(online_aggregated_observed_ratings, online_path, OBS)
        predicate_constructors.ratings_predicate(pd.concat(partitioned_truth_ratings[time_step:]),
                                                 online_path, TARGET, write_value=False)
        predicate_constructors.ratings_predicate(pd.concat(partitioned_truth_ratings[time_step:]),
                                                 online_path, TRUTH)

        # Target predicate.
        predicate_constructors.target_predicate(time_series_aggregated_truth_ratings, time_series_path, OBS)

        predicate_constructors.target_predicate(pd.concat(partitioned_truth_ratings[time_step:]),
                                                online_path, OBS)

        # Rated predicate.
        predicate_constructors.rated_predicate(time_series_aggregated_observed_ratings,
                                               time_series_aggregated_truth_ratings,
                                               time_series_path, OBS)

        predicate_constructors.rated_predicate(online_aggregated_observed_ratings,
                                               pd.concat(partitioned_truth_ratings[time_step:]),
                                               online_path, OBS)

        # Avg predicates.
        predicate_constructors.average_item_rating_predicate(time_series_aggregated_observed_ratings, time_series_path,
                                                             fold_unique_movies, fill_na=True)
        predicate_constructors.average_item_rating_predicate(online_aggregated_observed_ratings, online_path,
                                                             fold_unique_movies, fill_na=True)

        predicate_constructors.average_user_rating_predicate(time_series_aggregated_observed_ratings, time_series_path,
                                                             fold_unique_users, fill_na=True)
        predicate_constructors.average_user_rating_predicate(online_aggregated_observed_ratings, online_path,
                                                             fold_unique_users, fill_na=True)

    # Finally write commands file.
    time_series_command_list += ["STOP"]
    online_command_list += ["STOP"]
    command_file_write(time_series_command_list, time_series_out_directory)
    command_file_write(online_command_list, online_out_directory)


def construct_static_predicates(observed_ratings_df, truth_ratings_df, movies_df,
                                time_series_out_directory, online_out_directory):
    """
    Construct the predicates that do not change between timesteps.
    """
    # Set the shared path between these predicates.
    time_series_path = os.path.join(time_series_out_directory, "0".zfill(2))
    if not os.path.exists(time_series_path):
        os.makedirs(time_series_path)

    online_path = os.path.join(online_out_directory, "0".zfill(2))
    if not os.path.exists(online_path):
        os.makedirs(online_path)

    # Construct static predicates and save to time series path.
    predicate_constructors.nmf_ratings_predicate(observed_ratings_df, truth_ratings_df, time_series_path)
    predicate_constructors.sim_content_predicate(movies_df, time_series_path)
    predicate_constructors.sim_items_predicate(observed_ratings_df, time_series_path)
    predicate_constructors.sim_users_predicate(observed_ratings_df, time_series_path)

    # Construct static predicates copy static predicates to the online path.
    shutil.copy(os.path.join(time_series_path, "nmf_rating_obs.txt"),
                os.path.join(online_path, "nmf_rating_obs.txt"))
    shutil.copy(os.path.join(time_series_path, "sim_content_items_obs.txt"),
                os.path.join(online_path, "sim_content_items_obs.txt"))
    shutil.copy(os.path.join(time_series_path, "sim_items_obs.txt"),
                os.path.join(online_path, "sim_items_obs.txt"))
    shutil.copy(os.path.join(time_series_path, "sim_users_obs.txt"),
                os.path.join(online_path, "sim_users_obs.txt"))

    # Duplicate 0'th time step data for offline run on time steps > 0.
    for i in range(NUM_PARTITIONS - 1):
        # Set the shared path between these predicates.
        cp_time_series_path = os.path.join(time_series_out_directory, str(i + 1).zfill(2))
        if not os.path.exists(cp_time_series_path):
            os.makedirs(cp_time_series_path)

        cp_online_path = os.path.join(online_out_directory, str(i + 1).zfill(2))
        if not os.path.exists(cp_online_path):
            os.makedirs(cp_online_path)

        shutil.copy(os.path.join(time_series_path, "nmf_rating_obs.txt"),
                    os.path.join(cp_time_series_path, "nmf_rating_obs.txt"))
        shutil.copy(os.path.join(time_series_path, "nmf_rating_obs.txt"),
                    os.path.join(cp_online_path, "nmf_rating_obs.txt"))
        shutil.copy(os.path.join(time_series_path, "sim_content_items_obs.txt"),
                    os.path.join(cp_time_series_path, "sim_content_items_obs.txt"))
        shutil.copy(os.path.join(time_series_path, "sim_content_items_obs.txt"),
                    os.path.join(cp_online_path, "sim_content_items_obs.txt"))
        shutil.copy(os.path.join(time_series_path, "sim_items_obs.txt"),
                    os.path.join(cp_time_series_path, "sim_items_obs.txt"))
        shutil.copy(os.path.join(time_series_path, "sim_items_obs.txt"),
                    os.path.join(cp_online_path, "sim_items_obs.txt"))
        shutil.copy(os.path.join(time_series_path, "sim_users_obs.txt"),
                    os.path.join(cp_time_series_path, "sim_users_obs.txt"))
        shutil.copy(os.path.join(time_series_path, "sim_users_obs.txt"),
                    os.path.join(cp_online_path, "sim_users_obs.txt"))


def df_to_command(constants_df, value_series, action_type, partition_name, predicate_name):
    command_list = []
    assert(constants_df.shape[0] == value_series.shape[0])

    for idx, row in constants_df.iterrows():
        predicate_constants = row.values
        if value_series.loc[idx].shape[0] != 0:
            value = value_series.loc[idx].values[0]
        else:
            value = None
        command_list += [create_command_line(action_type, partition_name, predicate_name, predicate_constants, value)]
    return command_list


def create_command_line(action_type, partition_name, predicate_name, predicate_constants, value):
    if partition_name == OBS:
        partition_str = "READ"
    elif partition_name == TARGET:
        partition_str = "WRITE"

    quoted_predicate_constants = ["'" + str(const) + "'" for const in predicate_constants]
    constants_list = ",".join(quoted_predicate_constants)


    if action_type == ADD:
        if value is not None:
            return ADD + "\t" + partition_str + "\t" + predicate_name + "(" + constants_list + ")\t" + str(value)
        else:
            return ADD + "\t" + partition_str + "\t" + predicate_name + "(" + constants_list + ")"

    if action_type == OBSERVE:
        return OBSERVE + "\t" + predicate_name + "\t(" + constants_list + ")\t" + str(value)

    elif action_type == UPDATE:
        return UPDATE + "\t" + predicate_name + "(" + constants_list + ")\t" + str(value)

    elif action_type == DELETE:
        return DELETE + "\t" + partition_str + "\t" + predicate_name + "(" + constants_list + ")"


def command_file_write(command_list, path):
    command_str = ''
    for command in command_list:
        command_str += command + '\n'

    if not os.path.exists(path):
        os.makedirs(path)

    with open(os.path.join(path, 'commands.txt'), 'w') as writer:
        writer.write(command_str)


def sample_randomly(ratings_df, n_folds=N_FOLDS, sample_proportion=SAMPLE_PROPORTION):
    return [ratings_df.sample(frac=sample_proportion, random_state=i) for i in np.arange(n_folds)]


def partition(ratings_df, n_partitions=NUM_PARTITIONS):
    return np.array_split(ratings_df, n_partitions, axis=0)


def filter_dataframes(movies_df, ratings_df, n=0):
    """
    Get rid of users who have not yet rated more than n movies.
    """
    # filter ratings of movies without movie information
    filtered_ratings_df = ratings_df.loc[(slice(None), movies_df.index), :]
    print(filtered_ratings_df.head())

    # filter users that have less than n ratings
    filtered_ratings_df = filtered_ratings_df.groupby('userId').filter(lambda x: x.shape[0] > n)

    # filter movies in movie df
    filtered_movies_df = movies_df.loc[filtered_ratings_df.index.get_level_values('movieId').unique()]

    return filtered_movies_df, filtered_ratings_df


def load_dataframes():
    """
    Assuming that the raw data already exists in the data directory
    """
    movies_df = pd.read_csv(RAW_MOVIES_PATH, sep='::', header=None, encoding="ISO-8859-1", engine='python', skiprows=[0])
    movies_df.columns = ["movieId", "movie title", "genres"]
    movies_df = movies_df.join(movies_df["genres"].str.get_dummies('|')).drop('genres', axis=1)
    movies_df = movies_df.astype({'movieId': int})
    movies_df = movies_df.set_index('movieId')

    ratings_df = pd.read_csv(RAW_RATINGS_PATH, sep='::', header=None, engine='python', skiprows=[0])
    ratings_df.columns = ['userId', 'movieId', 'rating', 'timestamp']
    ratings_df = ratings_df.astype({'userId': int, 'movieId': int})
    ratings_df.rating = ratings_df.rating / ratings_df.rating.max()
    ratings_df = ratings_df.set_index(['userId', 'movieId'])

    return movies_df, ratings_df


def main():
    construct_predicates()


if __name__ == '__main__':
    main()
