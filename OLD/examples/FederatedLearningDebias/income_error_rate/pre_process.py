import argparse
import os
import shutil

import dill
import numpy as np
import pandas as pd

years = [2014, 2015, 2016, 2017, 2018]
states = [
    "AL",
    "AK",
    "AZ",
    "AR",
    "CA",
    "CO",
    "CT",
    "DE",
    "FL",
    "GA",
    "HI",
    "ID",
    "IL",
    "IN",
    "IA",
    "KS",
    "KY",
    "LA",
    "ME",
    "MD",
    "MA",
    "MI",
    "MN",
    "MS",
    "MO",
    "MT",
    "NE",
    "NV",
    "NH",
    "NJ",
    "NM",
    "NY",
    "NC",
    "ND",
    "OH",
    "OK",
    "OR",
    "PA",
    "RI",
    "SC",
    "SD",
    "TN",
    "TX",
    "UT",
    "VT",
    "VA",
    "WA",
    "WV",
    "WI",
    "WY",
    "PR",
]


def pre_process_income(data, years_list, states_list, group_name):
    """
    Pre-process the income dataset to make it ready for the simulation
    In this function we consider "SEX" as the sensitive value and "PINCP" as the target value.

    Args:
        data: the raw data
        years_list: the list of years to be considered
        states_list: the list of states to be considered

    Returns:
        Returns a list of pre-processed data for each state, if multiple years are
        selected, the data are concatenated.
        We return three lists:
        - The first list contains a pandas dataframe of features for each state
        - The second list contains a pandas dataframe of labels for each state
        - The third list contains a pandas dataframe of groups for each state
        The values in the list are numpy array of the dataframes
    """
    global years, states
    states = states_list if states_list is not None else states
    years = years_list if years_list is not None else years

    categorical_columns = ["COW", "SCHL", "MAR", "RAC1P"]
    continuous_columns = ["AGEP", "WKHP", "OCCP", "POBP", "RELP"]
    states = data.keys()
    dataframes = []
    labels = []
    groups = []
    for state in states:
        dataframe = pd.DataFrame()
        label = pd.DataFrame()
        group = pd.DataFrame()
        for year in years:
            year = str(year)
            df = data[state][year]["features_pd"]
            # convert the columns to one-hot encoding
            df = pd.get_dummies(df, columns=categorical_columns)

            # normalize the continuous columns between 0 and 1
            for col in continuous_columns:
                df[col] = (df[col] - df[col].min()) / (df[col].max() - df[col].min())
            data[state][year]["features_pd"] = df

            # convert label to int
            data[state][year]["labels_pd"] = data[state][year]["labels_pd"].astype(int)

            data[state][year]["groups_pd"][group_name] = [
                item  # 1 if item == 1 else 0
                for item in data[state][year]["groups_pd"][group_name]
            ]
            data[state][year]["features_pd"][group_name] = data[state][year]["groups_pd"][group_name]
            # concatenate the dataframes
            dataframe = pd.concat([dataframe, data[state][year]["features_pd"]])
            label = pd.concat([label, data[state][year]["labels_pd"]])
            group = pd.concat([group, data[state][year]["groups_pd"]])
        dataframes.append(dataframe.to_numpy())
        labels.append(label.to_numpy())
        groups.append(group.to_numpy())
    return dataframes, labels, groups


# todo: add the other datasets


parser = argparse.ArgumentParser(description="Flower Simulation with PyTorch")
parser.add_argument("--dataset_name", type=str, default=None)
parser.add_argument("--group", type=str, default=None)
parser.add_argument("--target", type=str, default=None)
parser.add_argument("--year", type=str, default=None, nargs="+")
parser.add_argument("--state", type=str, default=None, nargs="+")

args = parser.parse_args()
group = args.group
target = args.target
year = list(args.year) if args.year is not None else None
state = list(args.state) if args.state is not None else None

if args.dataset_name == "income":
    # read file
    dill_file = open("../data/income_data.pkd", "rb")
    data = dill.load(dill_file)
    dataframes, labels, groups = pre_process_income(
        data,
        years_list=year,
        states_list=state,
        group_name=group,
    )

    # save the pre-processed data
    for i in range(len(dataframes)):
        # create forlder for each state
        if os.path.exists(f"../data/income/{i}"):
            shutil.rmtree(f"../data/income/{i}")
        os.mkdir(f"../data/income/{i}")
        np.save(
            f"../data/income/{i}/income_dataframes_{i}.npy",
            dataframes[i],
        )
        np.save(
            f"../data/income/{i}/income_labels_{i}.npy",
            labels[i],
        )
        np.save(
            f"../data/income/{i}/income_groups_{i}.npy",
            groups[i],
        )
else:
    # Not implemeted yet for other datasets
    raise NotImplementedError
