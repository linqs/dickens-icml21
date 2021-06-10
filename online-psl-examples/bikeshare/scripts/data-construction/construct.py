"""
Construct PSL formatted data from raw bikehsare data.
"""

import datetime
import numpy as np
import os
import pandas as pd
import shutil

import predicate_constructors

DIRNAME = os.path.dirname(__file__)
DATA_PATH = os.path.join(DIRNAME, "../../data")
PSL_DATA_PATH = os.path.join(DIRNAME, "../../data/bikeshare")

OVERLAPPING_FOLDS = True
NUM_FOLDS = 10
NUM_PARTITIONS = 20
FOLD_SIZE = 1 / 3
INITIAL_PROPORTION = 1 / 3  # The proportion of the fold data that is initially observed for each fold.

# Command constants.
WRITE_INFERRED_COMMAND = "WRITEINFERREDPREDICATES"
ADD = 'ADDATOM'
OBSERVE = 'OBSERVEATOM'
UPDATE = 'UPDATEATOM'
DELETE = 'DELETEATOM'
CLOSE_COMMAND = 'STOP'
EXIT_COMMAND = 'EXIT'

# Partition names
OBS = 'obs'
TRUTH = 'truth'
TARGET = 'target'

FILE_INFO_MAP = {"ARIMA_obs.txt": ("ARIMA", OBS, False),
                 "Commute_obs.txt": ("Commute", OBS, False),
                 "Demand_obs.txt": ("Demand", OBS, True),
                 "IsHour_obs.txt": ("IsHour", OBS, False),
                 "IsWeekend_obs.txt": ("IsWeekend", OBS, False),
                 "IsDayOfWeek_obs.txt": ("IsDayOfWeek", OBS, False),
                 "Nearby_obs.txt": ("Nearby", OBS, False),
                 "Raining_obs.txt": ("Raining", OBS, False),
                 "Station_obs.txt": ("Station", OBS, False),
                 "Demand_target.txt": ("Demand", TARGET, True),
                 "Target_obs.txt": ("Target", TARGET, True)}


def construct_predicates():
    # Load the raw data.
    station_df, status_df, trip_df, weather_df = load_dataframes()

    # Partition dates into ranges for folds and splits.
    start_date = trip_df.start_date.min().date()
    end_date = trip_df.end_date.max().date()
    total_day_count = (end_date - start_date).days

    if OVERLAPPING_FOLDS:
        days_per_fold = np.floor(total_day_count * FOLD_SIZE)
    else:
        days_per_fold = np.floor(total_day_count / NUM_FOLDS)

    if OVERLAPPING_FOLDS:
        fold_offset = np.floor((total_day_count * (1 - FOLD_SIZE)) / (NUM_FOLDS - 1))
        fold_dates = [[start_date + datetime.timedelta(days=i * fold_offset),
                        start_date + datetime.timedelta(days=(i * fold_offset) + days_per_fold)] for i in range(NUM_FOLDS)]
    else:
        fold_dates = [[start_date + datetime.timedelta(days=i * days_per_fold),
                start_date + datetime.timedelta(days=(i + 1) * days_per_fold)] for i in range(NUM_FOLDS)]

    split_dates = [[]] * NUM_FOLDS
    for idx, split in enumerate(fold_dates):
        split_dates[idx] = [[split[0], split[0] + datetime.timedelta(days=INITIAL_PROPORTION * days_per_fold)]]
        days_per_split = np.floor((split[1] - split_dates[idx][0][1]).days / (NUM_PARTITIONS))
        split_dates[idx] += [[split_dates[idx][0][1] + datetime.timedelta(days=i * days_per_split),
                              split_dates[idx][0][1] + datetime.timedelta(days=(i + 1) * days_per_split)]
                             for i in range(NUM_PARTITIONS)]

    for fold in range(NUM_FOLDS):
        print("Constructing fold #" + str(fold))

        fold_status_df = status_df[(status_df.time.dt.date >= fold_dates[fold][0]) &
                                   (status_df.time.dt.date <= fold_dates[fold][1])]
        fold_trip_df = trip_df[(trip_df.end_date.dt.date >= fold_dates[fold][0]) &
                               (trip_df.end_date.dt.date <= fold_dates[fold][1])]
        fold_weather_df = weather_df[(weather_df.date.dt.date >= fold_dates[fold][0]) &
                                     (weather_df.date.dt.date <= fold_dates[fold][1])]

        # Construct demand dataframe for fold.
        # Demand is defined as the number trips starting at a stations over the that stations dock count.
        # Values are clipped to the range [0, 1].
        fold_trip_df_subset = fold_trip_df.loc[:, ['start_station_id', 'start_date', 'id']]
        fold_trip_df_subset.start_date = fold_trip_df_subset.start_date.dt.floor("60min")
        demand_df = fold_trip_df_subset.groupby(['start_station_id', 'start_date']).count().reset_index()
        demand_df.columns = ['station_id', 'time', 'demand']
        demand_df.demand = np.clip(demand_df.demand / station_df.loc[demand_df.station_id, 'dock_count'].values,
                                   0.0, 1.0)

        # Fill in missing data ranges.
        demand_df = demand_df.append(pd.DataFrame(data={'station_id': -1, 'time': status_df.time.unique(), 'demand': 0}))
        demand_df = demand_df.set_index(['station_id', 'time']).unstack(fill_value=0).stack().reset_index()
        demand_df.drop(demand_df[demand_df.station_id == -1].index, inplace=True)

        # Instantiate predicate constructor object.
        predicate_constructor = predicate_constructors.predicate_constructor(demand_df)
        out_directory = PSL_DATA_PATH + '/bikeshare_time_series/' + str(int(fold)).zfill(2) + '/eval/'

        print("Making " + out_directory)
        if not os.path.exists(out_directory):
            os.makedirs(out_directory)

        # Construct predicates that are static for this fold.
        print("Constructing static predicates.")
        construct_static_predicates(predicate_constructor, station_df, fold_weather_df, fold_status_df, fold_trip_df,
                                    split_dates[fold], out_directory)

        # Construct predicates that are dynamic for this fold.
        print("Constructing dynamic predicates.")
        construct_dynamic_predicates(predicate_constructor, station_df, fold_weather_df, fold_status_df, fold_trip_df,
                                demand_df, split_dates[fold], out_directory)


def construct_client_commands(predicate_constructor,
                              prev_observed_demands, current_observed_demands,
                              prev_truth_demands, current_truth_demands,
                              prev_observed_raining, current_observed_raining,
                              prev_observed_ARIMA, current_observed_ARIMA,
                              time_step):
    add_targets_command_list = []
    add_observation_command_list = []
    observe_command_list = []
    extra_commands = [WRITE_INFERRED_COMMAND + "\t'./inferred-predicates/{:02d}'".format(time_step)]

    if time_step > 0:
        # Demand.
        new_targets_df = current_truth_demands.loc[current_truth_demands.index.difference(prev_truth_demands.index)].reset_index()
        add_targets_command_list += df_to_command(
            predicate_constructor.time_to_int_ids(new_targets_df.loc[:, ['station_id', 'time']]),
            new_targets_df.loc[:, []],
            ADD, TARGET, 'Demand')

        new_observed_demands_df = current_observed_demands.loc[
            current_observed_demands.index.difference(prev_observed_demands.index)].reset_index()
        observe_command_list += df_to_command(
            predicate_constructor.time_to_int_ids(new_observed_demands_df.loc[:, ['station_id', 'time']]),
            new_observed_demands_df.loc[:, ['demand']],
            OBSERVE, OBS, 'Demand')

        # Target.
        new_target_df = current_truth_demands.loc[
            current_truth_demands.index.difference(prev_truth_demands.index)].reset_index()
        add_observation_command_list += df_to_command(
            predicate_constructor.time_to_int_ids(new_target_df.loc[:, ['station_id', 'time']]),
            new_target_df.loc[:, ['demand']].clip(1, 1),
            ADD, OBS, 'Target')

        # Raining.
        new_observed_raining_df = current_observed_raining.loc[
            current_observed_raining.index.difference(prev_observed_raining.index)].reset_index()
        add_observation_command_list += df_to_command(
            predicate_constructor.time_to_int_ids(new_observed_raining_df.loc[:, ['station_id', 'time']]),
            new_observed_raining_df.loc[:, ['raining']].clip(1, 1),
            ADD, OBS, 'Raining')

        # ARIMA.
        new_observed_ARIMA_df = current_observed_ARIMA.loc[
            current_observed_ARIMA.index.difference(prev_observed_ARIMA.index)].reset_index()
        add_observation_command_list += df_to_command(
            predicate_constructor.time_to_int_ids(new_observed_ARIMA_df.loc[:, ['time']]),
            new_observed_ARIMA_df.loc[:, ['ARIMA_Predictions']].clip(1, 1),
            ADD, OBS, 'ARIMA')

    command_list = (add_targets_command_list + observe_command_list +
                    add_observation_command_list + extra_commands)

    return command_list


def construct_dynamic_predicates(predicate_constructor, station_df, weather_df, status_df, trip_df, demand_df,
                                 split_dates, out_directory):
    """
    Construct the predicates change between timesteps.
    """

    path = out_directory

    # Initialize observations.
    # This dataframe will grow with each timestep by adding some or all of the previous timestep's target set.
    aggregated_observed_demand_df = pd.DataFrame(columns=demand_df.columns)
    aggregated_observed_demand_df = aggregated_observed_demand_df.append(
        demand_df[(demand_df["time"].dt.date > split_dates[0][0]) &
                  (demand_df["time"].dt.date < split_dates[0][1])])
    target_demands_df = demand_df[(demand_df["time"].dt.date > split_dates[1][0]) &
                                  (demand_df["time"].dt.date < split_dates[1][1])]

    # Initialize empty list of commands.
    command_list = []

    for time_step, split_date_range in enumerate(split_dates[:-1]):
        print("Constructing predicates for time step: " + str(time_step).zfill(2))
        # Set the shared path between these predicates.
        path = os.path.join(out_directory, str(time_step).zfill(2))
        if not os.path.exists(path):
            os.makedirs(path)

        if time_step == 0:
            # Initialize raining and ARIMA dfs for 0th timestep.
            # Raining
            raining_df = predicate_constructor.raining_predicate(weather_df, station_df, target_demands_df, path)

            # Arima
            ARIMA_df = predicate_constructor.arima_predicate(aggregated_observed_demand_df, target_demands_df, path)

        # Update timestep observations and targets.
        prev_observed_demand_df = aggregated_observed_demand_df.copy()
        prev_target_demand_df = target_demands_df.copy()
        prev_raining_df = raining_df.copy()
        prev_ARIMA_df = ARIMA_df.copy()

        if time_step > 0:
            aggregated_observed_demand_df = aggregated_observed_demand_df.append(
                demand_df[(demand_df["time"].dt.date > split_dates[time_step][0]) &
                          (demand_df["time"].dt.date < split_dates[time_step][1])])
            target_demands_df = demand_df[(demand_df["time"].dt.date > split_dates[time_step + 1][0]) &
                                      (demand_df["time"].dt.date < split_dates[time_step + 1][1])]

        # Demand
        predicate_constructor.demand_predicate(aggregated_observed_demand_df, path, OBS)
        predicate_constructor.demand_predicate(target_demands_df, path, TARGET, write_value=False)
        predicate_constructor.demand_predicate(target_demands_df, path, TRUTH)

        # Target
        predicate_constructor.target_predicate(aggregated_observed_demand_df.append(target_demands_df), path, OBS)

        # Arima
        ARIMA_df = predicate_constructor.arima_predicate(aggregated_observed_demand_df, target_demands_df, path)

        # Raining
        raining_df = predicate_constructor.raining_predicate(weather_df, station_df, target_demands_df, path)


        # Get client commands for this timestep.
        command_list += construct_client_commands(predicate_constructor,
                                                  prev_observed_demand_df, aggregated_observed_demand_df,
                                                  prev_target_demand_df, target_demands_df,
                                                  prev_raining_df, raining_df,
                                                  prev_ARIMA_df, ARIMA_df,
                                                  time_step)

    command_list += ["STOP"]
    command_file_write(command_list, out_directory)


def construct_static_predicates(predicate_constructor, station_df, weather_df, status_df, trip_df, split_dates, out_directory):
    """
    Construct the predicates that do not change between timesteps.
    """
    # Set the shared path between these predicates.
    path = os.path.join(out_directory, "0".zfill(2))
    if not os.path.exists(path):
        os.makedirs(path)

    # Get observed trips for initial split of fold.
    obs_trip_df = trip_df[trip_df["end_date"].dt.date < split_dates[0][1]]

    # Time related predicates.
    # Status is only used for time information in these predicates.
    predicate_constructor.ishour_predicate(status_df.loc[:, ["station_id", "time"]], path)
    predicate_constructor.isdayofweek_predicate(status_df.loc[:, ["station_id", "time"]], path)
    predicate_constructor.isweekend_predicate(status_df.loc[:, ["station_id", "time"]], path)

    # Location / station related predicates.
    predicate_constructor.station_predicate(station_df, path)
    predicate_constructor.nearby_predicate(station_df, path)
    predicate_constructor.commute_predicate(obs_trip_df, path)

    # Duplicate 0'th time step data for offline run on time steps > 0.
    for i in range(NUM_PARTITIONS - 1):
        # Set the shared path between these predicates.
        cp_path = os.path.join(out_directory, str(i + 1).zfill(2))
        if not os.path.exists(cp_path):
            os.makedirs(cp_path)

        shutil.copy(os.path.join(path, "IsHour_obs.txt"), os.path.join(cp_path, "IsHour_obs.txt"))
        shutil.copy(os.path.join(path, "IsDayOfWeek_obs.txt"), os.path.join(cp_path, "IsDayOfWeek_obs.txt"))
        shutil.copy(os.path.join(path, "IsWeekend_obs.txt"), os.path.join(cp_path, "IsWeekend_obs.txt"))

        shutil.copy(os.path.join(path, "Station_obs.txt"), os.path.join(cp_path, "Station_obs.txt"))
        shutil.copy(os.path.join(path, "Nearby_obs.txt"), os.path.join(cp_path, "Nearby_obs.txt"))
        shutil.copy(os.path.join(path, "Commute_obs.txt"), os.path.join(cp_path, "Commute_obs.txt"))


def command_file_write(command_list, path):
    command_str = ''
    for command in command_list:
        command_str += command + '\n'

    if not os.path.exists(path):
        os.makedirs(path)

    with open(os.path.join(path, 'commands.txt'), 'w') as writer:
        writer.write(command_str)


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
        return OBSERVE + "\t" + predicate_name + "(" + constants_list + ")\t" + str(value)

    elif action_type == UPDATE:
        return UPDATE + "\t" + predicate_name + "(" + constants_list + ")\t" + str(value)

    elif action_type == DELETE:
        return DELETE + "\t" + partition_str + "\t" + predicate_name + "(" + constants_list + ")"


def load_dataframes():
    station_df = pd.read_csv(DATA_PATH + "/bikeshare_raw/station.csv", sep=',', encoding="ISO-8859-1", engine='python')
    station_df = station_df.set_index('id')

    # Status df contains data about the station on a minute frequency.
    status_df = pd.read_csv(DATA_PATH + "/bikeshare_raw/status.csv", sep=',', encoding="ISO-8859-1", engine='python',
            parse_dates=['time'], infer_datetime_format=True)
    # Aggregate status entries to the hour.
    status_df.time = status_df.time.dt.floor('60min')
    status_df = status_df.groupby(['station_id', 'time']).mean().reset_index()

    trip_df = pd.read_csv(DATA_PATH + "/bikeshare_raw/trip.csv", sep=',', encoding="ISO-8859-1", engine='python',
                          parse_dates=['start_date', 'end_date'], infer_datetime_format=True)

    weather_df = pd.read_csv(DATA_PATH + "/bikeshare_raw/weather.csv", sep=',', encoding="ISO-8859-1", engine='python',
                             parse_dates=['date'], infer_datetime_format=True)

    # filter status and station that do not exist in early trip_df
    station_df = station_df[:-2]
    status_df = status_df[status_df.station_id.isin(station_df.index.unique())]
    trip_df = trip_df[trip_df.start_station_id.isin(station_df.index.unique())]
    trip_df = trip_df[trip_df.end_station_id.isin(station_df.index.unique())]

    return station_df, status_df, trip_df, weather_df


def main():
    construct_predicates()


if __name__ == '__main__':
    main()
