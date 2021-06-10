"""
Define class with methods for constructing and writing PSL predicates from raw bikeshare data.
"""

import numpy as np
import os
import pandas as pd
import pgeocode
import time

from multiprocessing import Pool
from sklearn.metrics.pairwise import haversine_distances
from statsmodels.tsa.statespace.sarimax import SARIMAX



class predicate_constructor:
    time_to_constant_dict = {}

    def __init__(self, demand_df):
        self.time_to_constant_dict = dict(zip(np.sort(demand_df.time.unique()),
                                              np.arange(len(demand_df.time.unique()), dtype=int)))

    def time_to_int_ids(self, data):
        data_copy = data.copy(deep=True)

        # Convert times to int ids.
        for col in data_copy.select_dtypes(include=np.datetime64).columns:
            data_copy[col] = data_copy[col].map(self.time_to_constant_dict)

        return data_copy

    def write(self, data, predicate_name, path):
        data_copy = self.time_to_int_ids(data)

        # Path to this file relative to caller
        data_copy.to_csv(os.path.join(path, predicate_name + '.txt'), sep='\t', header=False, index=False)

    def ishour_predicate(self, status_df, path):
        unique_times = status_df.time.unique()
        ishour_df = pd.DataFrame(data={'time': unique_times,
                                       'hour': [pd.Timestamp(t).hour for t in unique_times],
                                       'value': 1})
        self.write(ishour_df, "IsHour_obs", path)

    def isdayofweek_predicate(self, status_df, path):
        unique_times = status_df.time.unique()
        sameweekday_df = pd.DataFrame(data={'time': unique_times,
                                            'day': [pd.Timestamp(t).dayofweek for t in unique_times],
                                            'value': 1})
        self.write(sameweekday_df, "IsDayOfWeek_obs", path)

    def isweekend_predicate(self, status_df, path):
        unique_times = status_df.time.unique()
        isweekend_df = pd.DataFrame(data={'time': unique_times,
                                          'isWeekend': [np.floor(pd.Timestamp(t).dayofweek / 5) for t in unique_times]})
        self.write(isweekend_df, "IsWeekend_obs", path)

    def station_predicate(self, station_df, path):
        self.write(pd.DataFrame(data={'station': station_df.index,
                                      'value': 1}), "Station_obs", path)

    def nearby_predicate(self, station_df, path, n=5):
        distances_df = pd.DataFrame(data=haversine_distances(station_df.loc[:, ['lat', 'long']]),
                                    index=station_df.index, columns=station_df.index)

        # Keep top 5 closest for each station
        # take top n for each movie to define pairwise blocks
        top_n_frame = pd.DataFrame(index=station_df.index, columns=range(5))
        for m in station_df.index:
            top_n_frame.loc[m, :] = distances_df.loc[m].nsmallest(n).index

        flattened_frame = top_n_frame.values.flatten()
        distance_index = np.array([[i] * n for i in station_df.index]).flatten()
        distance_index = pd.MultiIndex.from_arrays([distance_index, flattened_frame])
        nearby_series = pd.Series(data=1, index=distance_index)

        self.write(nearby_series.reset_index(), 'Nearby_obs', path)

    def commute_predicate(self, trip_df, path, min_threshold=40, ratio_threshold=0.4):

        # Filter trips to those between 6am and 7pm on weekdays (common work hours).
        filtered_trip_df = trip_df[(trip_df.start_date.dt.hour > 6) &
                                   (trip_df.start_date.dt.hour < 19) &
                                   (trip_df.start_date.dt.day > 4)]

        # Count the number of relevant trips between stations.
        trip_count_df = filtered_trip_df.loc[:, ["id", "start_station_id", "end_station_id"]].groupby(
            ["start_station_id", "end_station_id"]).count()
        trip_count_df.columns = ["count"]

        # Fill return count
        trip_count_df = trip_count_df.reindex(trip_count_df.index.union(trip_count_df.swaplevel().index), fill_value=0.0)
        trip_count_df.loc[:, "return_count"] = trip_count_df.loc[trip_count_df.swaplevel().index, "count"].values

        # Filter out station pairs that do not meet commute thresholds.
        commute_routes = trip_count_df[(trip_count_df.loc[:, "count"] > min_threshold) &
                                       (trip_count_df.loc[:, "return_count"] > min_threshold) &
                                       (pd.DataFrame({1: (trip_count_df.loc[:, "count"] / trip_count_df.loc[:, "return_count"]),
                                                      2: (trip_count_df.loc[:, "return_count"] / trip_count_df.loc[:, "count"])}
                                                     ).min(axis=1) > ratio_threshold)].reset_index().loc[:, ["level_0", "level_1"]]

        commute_routes.loc[:, "value"] = 1.0
        self.write(commute_routes, "Commute_obs", path)

    def demand_predicate(self, demand_df, path, partition, write_value=True):
        if write_value:
            self.write(demand_df, 'Demand_{}'.format(partition), path)
        else:
            self.write(demand_df[['station_id', 'time']], 'Demand_{}'.format(partition), path)

    def target_predicate(self, demand_df, path, partition):
        # truth
        target_dataframe = demand_df[['station_id', 'time']]
        target_dataframe['value'] = 1
        self.write(target_dataframe, 'Target_{}'.format(partition), path)

    def arima_predicate(self, original_obs_demand_df, original_target_demand_df, path):
        obs_demand_df = original_obs_demand_df.set_index(['station_id', 'time'])

        avg_demand_series = obs_demand_df.groupby('time').mean()
        predicted_demand_df = pd.DataFrame(index=original_target_demand_df.time.unique())

        arima_model = SARIMAX(avg_demand_series.values, order=(0, 0, 0), seasonal_order=(1, 1, 1, 24))
        fitted_arima_model = arima_model.fit(disp=False)
        arima_predictions = np.clip(fitted_arima_model.forecast(steps=predicted_demand_df.shape[0]), 0, 1)
        predicted_demand_df.loc[predicted_demand_df.index, 'ARIMA_predictions'] = arima_predictions
        predicted_demand_df = predicted_demand_df.reset_index()
        predicted_demand_df.columns = ['time', 'ARIMA_Predictions']

        self.write(predicted_demand_df.loc[:, ['time', 'ARIMA_Predictions']], 'ARIMA_obs', path)
        return predicted_demand_df.loc[:, ['time', 'ARIMA_Predictions']]

    def station_to_zipcode_map(self, station_df, weather_df):
        zipcode_list = weather_df["zip_code"].unique()
        zipmodel = pgeocode.Nominatim('us')
        zip_to_station = dict({})

        for idx, row in station_df.iterrows():
            zip_distances = []

            for zipcode in zipcode_list:
                zipcode_info = zipmodel.query_postal_code(str(zipcode))
                zip_lat = zipcode_info["latitude"]
                zip_lon = zipcode_info["longitude"]

                zip_distances += [(zipcode, haversine_distances([[row["lat"], row["long"]]],
                                                                [[zip_lat, zip_lon]])[0][0])]

            nearest_zip = sorted(zip_distances, key=lambda x: x[1])[0][0]

            if nearest_zip not in zip_to_station.keys():
                zip_to_station[nearest_zip] = [idx]
            else:
                zip_to_station[nearest_zip] += [idx]

        return zip_to_station

    def raining_predicate(self, weather_df, station_df, demand_df, path):
        raining_df = pd.DataFrame({'station_id': demand_df.station_id,
                                   'time': demand_df.time,
                                   'raining': 0})

        zip_to_station = self.station_to_zipcode_map(station_df, weather_df)
        weather_events_df = weather_df[weather_df.events.notnull()]

        for idx, row in weather_events_df.iterrows():
            for station_id in zip_to_station[row["zip_code"]]:
                raining_df[(raining_df.time.dt.floor("1440min") == row["date"])
                           & (raining_df.station_id == station_id)].raining = 1

        self.write(raining_df, "Raining_obs", path)

        return raining_df

