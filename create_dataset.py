import argparse

import dill
import folktables
import numpy as np
import pandas as pd
from folktables import ACSDataSource, ACSEmployment, ACSIncome, ACSTravelTime


def create_employment_dataset():
    ACSEmployment = folktables.BasicProblem(
        features=[
            "AGEP",
            "SCHL",
            "MAR",
            "RELP",
            "DIS",
            "ESP",
            "CIT",
            "MIG",
            "MIL",
            "ANC",
            "NATIVITY",
            "DEAR",
            "DEYE",
            "DREM",
            "SEX",
            "RAC1P",
        ],
        target="ESR",
        target_transform=lambda x: x == 1,
        group="SEX",
        preprocess=lambda x: x,
        postprocess=lambda x: np.nan_to_num(x, -1),
    )

    data = {}

    state_list = [
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
    for year in [2014, 2015, 2016, 2017, 2018, 2019]:
        for state in state_list:
            data_source = ACSDataSource(
                survey_year=year, horizon="1-Year", survey="person"
            )
            acs_data = data_source.get_data(states=[state], download=True)
            try:
                features, label, group = ACSEmployment.df_to_numpy(acs_data)
                feature_pd, label_pd, group_pd = ACSEmployment.df_to_pandas(acs_data)
                print("Downloaded state: ", state, " and year: ", year)
                if state not in data:
                    data[state] = {}
                    data[state]["features"] = features
                    data[state]["labels"] = label
                    data[state]["groups"] = group
                    data[state]["features_pd"] = feature_pd
                    data[state]["labels_pd"] = label_pd
                    data[state]["groups_pd"] = group_pd
                else:
                    data[state]["features"] = np.concatenate(
                        (data[state]["features"], features), axis=0
                    )
                    data[state]["labels"] = np.concatenate(
                        (data[state]["labels"], label), axis=0
                    )
                    data[state]["groups"] = np.concatenate(
                        (data[state]["groups"], group), axis=0
                    )
                    data[state]["features_pd"] = pd.concat(
                        (data[state]["features_pd"], feature_pd), axis=0
                    )
                    data[state]["labels_pd"] = pd.concat(
                        (data[state]["labels_pd"], label_pd), axis=0
                    )
                    data[state]["groups_pd"] = pd.concat(
                        (data[state]["groups_pd"], group_pd), axis=0
                    )
            except:
                print("Error with state: ", state, " and year: ", year)

    return data


def adult_filter(data):
    """Mimic the filters in place for Adult data.

    Adult documentation notes: Extraction was done by Barry Becker from
    the 1994 Census database. A set of reasonably clean records was extracted
    using the following conditions:
    ((AAGE>16) && (AGI>100) && (AFNLWGT>1)&& (HRSWK>0))
    """
    df = data
    df = df[df["AGEP"] > 16]
    df = df[df["PINCP"] > 100]
    df = df[df["WKHP"] > 0]
    df = df[df["PWGTP"] >= 1]
    return df


def create_income_dataset():
    ACSIncome = folktables.BasicProblem(
        features=[
            "AGEP",
            "COW",
            "SCHL",
            "MAR",
            "OCCP",
            "POBP",
            "RELP",
            "WKHP",
            "SEX",
            "RAC1P",
        ],
        target="PINCP",
        target_transform=lambda x: x > 50000,
        group="SEX",
        preprocess=adult_filter,
        postprocess=lambda x: np.nan_to_num(x, -1),
    )

    data = {}

    state_list = [
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
    for year in [2014, 2015, 2016, 2017, 2018, 2019]:
        for state in state_list:
            data_source = ACSDataSource(
                survey_year=year, horizon="1-Year", survey="person"
            )
            acs_data = data_source.get_data(states=[state], download=True)
            try:
                features, label, group = ACSIncome.df_to_numpy(acs_data)
                feature_pd, label_pd, group_pd = ACSIncome.df_to_pandas(acs_data)
                print("Downloaded state: ", state, " and year: ", year)
                if state not in data:
                    data[state] = {}
                    data[state]["features"] = features
                    data[state]["labels"] = label
                    data[state]["groups"] = group
                    data[state]["features_pd"] = feature_pd
                    data[state]["labels_pd"] = label_pd
                    data[state]["groups_pd"] = group_pd
                else:
                    data[state]["features"] = np.concatenate(
                        (data[state]["features"], features), axis=0
                    )
                    data[state]["labels"] = np.concatenate(
                        (data[state]["labels"], label), axis=0
                    )
                    data[state]["groups"] = np.concatenate(
                        (data[state]["groups"], group), axis=0
                    )
                    data[state]["features_pd"] = pd.concat(
                        (data[state]["features_pd"], feature_pd), axis=0
                    )
                    data[state]["labels_pd"] = pd.concat(
                        (data[state]["labels_pd"], label_pd), axis=0
                    )
                    data[state]["groups_pd"] = pd.concat(
                        (data[state]["groups_pd"], group_pd), axis=0
                    )
            except:
                print("Error with state: ", state, " and year: ", year)

    return data


data = create_income_dataset()
dill.dump(data, open("income_data.pkd", "wb"))


def __main__():
    parser = argparse.ArgumentParser(description="Flower Simulation with PyTorch")
    parser.add_argument("--dataset_name", type=str, default=None)
    parser.add_argument("--group", type=str, default=None)
    parser.add_argument("--target", type=str, default=None)
    parser.add_argument("--year", type=str, default=None)
    parser.add_argument("--state", type=str, default=None)

    args = parser.parse_args()

    assert args.dataset_name is not None, "Please specify a dataset name"

    if args.dataset_name == "income":
        data = create_income_dataset()
        dill.dump(data, open("../data/income_data.pkd", "wb"))

    elif args.dataset_name == "employment":
        data = create_employment_dataset()
        dill.dump(data, open("../data/employment_data.pkd", "wb"))
    else:
        raise Exception("Invalid dataset name")
