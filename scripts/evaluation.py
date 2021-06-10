import os
import re
import sys
import pandas as pd
import numpy as np


from scipy.stats import pearsonr
from math import sqrt

def pearson_correlation(prediction_list, truth_list):
    return pearsonr(prediction_list, truth_list)[0]


def rmse(prediction_list, truth_list):
    rmse = 0
    for prediction, truth in zip(prediction_list, truth_list):
        rmse += (float(prediction) - float(truth))**2

    return sqrt(rmse / len(truth_list))


def mrr(truth_dict, pred_dict):
    df_columns = ['user_id', 'movie_id', 'rating']
    pred_df = pd.Series(pred_dict).reset_index()

    pred_df.columns = df_columns

    # create truth dict for just this iteration
    fold_truth_dict = dict({})
    for key in pred_dict.keys():
        if key not in truth_dict.keys():
            continue
        fold_truth_dict[key] = truth_dict[key]

    truth_df = pd.Series(fold_truth_dict).reset_index()
    truth_df.columns = df_columns

    mrr = 0
    user_count = len(truth_df.user_id.unique())
    for user in truth_df.user_id.unique():
        user_truth_ratings = truth_df.loc[truth_df.user_id == user].sort_values(by=['rating'], ascending=False).reset_index()
        user_pred_ratings = pred_df.loc[pred_df.user_id == user].sort_values(by=['rating'], ascending=False).reset_index()

        most_recommended = user_truth_ratings.loc[user_truth_ratings.rating == user_truth_ratings.rating.max()].movie_id.values
        pred_rank = user_pred_ratings.loc[user_pred_ratings.movie_id.isin(most_recommended)].index[0] + 1

        mrr += float(1 / pred_rank)
    return float(mrr / user_count)


def create_lists(predictions_path, truth_dict):
    truth_list = []
    prediction_list = []

    with open(predictions_path, 'r') as results_file:
        for line in results_file:
            parts = line.strip().split()

            if tuple(parts[:-1]) not in truth_dict:
                continue
            truth_list.append(truth_dict[tuple(parts[:-1])])
            prediction_list.append(float(parts[-1]))

    return prediction_list, truth_list


def load_preds(pred_file):
    pred_dict = dict({})
    with open(pred_file, 'r') as results_file:
        for line in results_file:
            parts = line.strip().split()
            pred_dict[tuple(parts[:-1])] = float(parts[-1])
    return pred_dict


def load_truth(data_path):
    truth_dict = {}

    if not os.path.isdir(data_path):
        return None

    for fold in os.listdir(data_path):
        eval_path = os.path.join(data_path, fold, 'eval')
        if not os.path.isdir(eval_path):
            continue

        for split in range(len(os.listdir(eval_path))):
            if not os.path.isdir(os.path.join(eval_path, str(split))):
                continue

            for psl_file in os.listdir(os.path.join(eval_path, str(split))):
                if 'truth' not in psl_file:
                    continue
                with open(os.path.join(eval_path, str(split), psl_file), 'r') as results_file:
                    for line in results_file:
                        parts = line.strip().split()
                        truth_dict[tuple(parts[:-1])] = float(parts[-1])

    return truth_dict
