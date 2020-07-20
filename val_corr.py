""" Example of how to calculate validation correlation on your predictions"""


import numpy as np
import pandas as pd


# Scoring function stripped from Numerai's example_model
def score(df):
    """Ranks Predictions and returns correlation"""
    pct_ranks = df[PREDICTION_NAME].rank(pct=True, method="first")
    targets = df[TARGET_NAME]
    return np.corrcoef(targets, pct_ranks)[0, 1]


# CONSTANTS
ROUND = '221'
TOURNAMENT_NAME = "kazutsugi"
TARGET_NAME = f"target_{TOURNAMENT_NAME}"
PREDICTION_NAME = f"prediction_{TOURNAMENT_NAME}"
submission = "Numerai/YourSubmissionName.csv"


tournament_data = pd.read_csv("Numerai/Datasets/numerai_dataset_" +
                              ROUND + "/numerai_tournament_data.csv").set_index("id")


submission_data = pd.read_csv(submission).set_index("id")
submission_data['era'] = tournament_data['era']
submission_data['data_type'] = tournament_data['data_type']
submission_data[TARGET_NAME] = tournament_data[TARGET_NAME]

# print(submission_data)

validation_data = submission_data[submission_data.data_type == "validation"]
# print(validation_data)

validation_correlations = validation_data.groupby("era").apply(score)
# print(validation_correlations)

# This is the validation correlation from the site:
print(validation_correlations.mean())
