import argparse

import dill
import folktables
import numpy as np
import pandas as pd
from folktables import ACSDataSource, ACSEmployment, ACSIncome, ACSTravelTime

years = [2014, 2015, 2016, 2017, 2018]
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


def travel_time_filter(data):
    """
    Filters for the employment prediction task
    """
    df = data
    df = df[df["AGEP"] > 16]
    df = df[df["PWGTP"] >= 1]
    df = df[df["ESR"] == 1]
    return df


def public_coverage_filter(data):
    """
    Filters for the public health insurance prediction task; focus on low income Americans, and those not eligible for Medicare
    """
    df = data
    df = df[df["AGEP"] < 65]
    df = df[df["PINCP"] <= 30000]
    return df


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


def create_employment_dataset(year: list, state: list, group: str, target: str):
    """
    Downloads the US Census ACS data for the given year and state
    and creates a ACSEmployment dataset

    If year is None, downloads all years
    If state is None, downloads all states
    If group is not specified, the default sensitive value will be "SEX"
    If target is not specified, the default target value will be "ESR"

    Returns a dictionary in which the keys are the state names and the values are
    dictionaries where the keys are the years. For each year we will have a dictionary
    with the following keys:
        - features: numpy array of features
        - labels: numpy array of labels
        - groups: numpy array of groups
        - features_pd: pandas dataframe of features
        - labels_pd: pandas dataframe of labels
        - groups_pd: pandas dataframe of groups
    Example:
        {
            "AL": {
                "2014": {
                    "features": np.array([]),
                    "labels": np.array([]),
                    "groups": np.array([]),
                    "features_pd": pd.DataFrame([]),
                    "labels_pd": pd.DataFrame([]),
                    "groups_pd": pd.DataFrame([]),
                },
            }
        }
    """
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
        target="ESR" if target is None else target,
        target_transform=lambda x: x == 1,
        group="SEX" if group is None else group,
        preprocess=lambda x: x,
        postprocess=lambda x: np.nan_to_num(x, -1),
    )

    data = create_dataset(year_list=year, state=state, ACSDataset=ACSEmployment)
    return data


def create_income_dataset(year: list, state: list, group: str, target: str):
    """
    Downloads the US Census ACS data for the given year and state
    and creates the ACSIncome dataset

    If year is None, downloads all years
    If state is None, downloads all states
    If group is not specified, the default sensitive value will be "SEX"
    If target is not specified, the default target value will be "PINCP"

    Returns a dictionary in which the keys are the state names and the values are
    dictionaries where the keys are the years. For each year we will have a dictionary
    with the following keys:
        - features: numpy array of features
        - labels: numpy array of labels
        - groups: numpy array of groups
        - features_pd: pandas dataframe of features
        - labels_pd: pandas dataframe of labels
        - groups_pd: pandas dataframe of groups
    Example:
        {
            "AL": {
                "2014": {
                    "features": np.array([]),
                    "labels": np.array([]),
                    "groups": np.array([]),
                    "features_pd": pd.DataFrame([]),
                    "labels_pd": pd.DataFrame([]),
                    "groups_pd": pd.DataFrame([]),
                },
            }
        }
    """
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
        target="PINCP" if target is None else target,
        target_transform=lambda x: x > 50000,
        group="SEX" if group is None else group,
        preprocess=adult_filter,
        postprocess=lambda x: np.nan_to_num(x, -1),
    )

    data = create_dataset(year_list=year, state=state, ACSDataset=ACSIncome)
    return data


def create_travel_time_dataset(year: list, state: list, group: str, target: str):
    """
    Downloads the US Census ACS data for the given year and state
    and creates the ACSTravelTime dataset

    If year is None, downloads all years
    If state is None, downloads all states
    If group is not specified, the default sensitive value will be "SEX"
    If target is not specified, the default target value will be "PINCP"

    Returns a dictionary in which the keys are the state names and the values are
    dictionaries where the keys are the years. For each year we will have a dictionary
    with the following keys:
        - features: numpy array of features
        - labels: numpy array of labels
        - groups: numpy array of groups
        - features_pd: pandas dataframe of features
        - labels_pd: pandas dataframe of labels
        - groups_pd: pandas dataframe of groups
    Example:
        {
            "AL": {
                "2014": {
                    "features": np.array([]),
                    "labels": np.array([]),
                    "groups": np.array([]),
                    "features_pd": pd.DataFrame([]),
                    "labels_pd": pd.DataFrame([]),
                    "groups_pd": pd.DataFrame([]),
                },
            }
        }
    """
    ACSTravelTime = folktables.BasicProblem(
        features=[
            "AGEP",
            "SCHL",
            "MAR",
            "SEX",
            "DIS",
            "ESP",
            "MIG",
            "RELP",
            "RAC1P",
            "PUMA",
            "ST",
            "CIT",
            "OCCP",
            "JWTR",
            "POWPUMA",
            "POVPIP",
        ],
        target="JWMNP",
        target_transform=lambda x: x > 20,
        group="RAC1P",
        preprocess=travel_time_filter,
        postprocess=lambda x: np.nan_to_num(x, -1),
    )

    data = create_dataset(year_list=year, state=state, ACSDataset=ACSTravelTime)
    return data


def create_mobility_dataset(year: list, state: list, group: str, target: str):
    """
    Downloads the US Census ACS data for the given year and state
    and creates the mobility dataset

    If year is None, downloads all years
    If state is None, downloads all states
    If group is not specified, the default sensitive value will be "SEX"
    If target is not specified, the default target value will be "PINCP"

    Returns a dictionary in which the keys are the state names and the values are
    dictionaries where the keys are the years. For each year we will have a dictionary
    with the following keys:
        - features: numpy array of features
        - labels: numpy array of labels
        - groups: numpy array of groups
        - features_pd: pandas dataframe of features
        - labels_pd: pandas dataframe of labels
        - groups_pd: pandas dataframe of groups
    Example:
        {
            "AL": {
                "2014": {
                    "features": np.array([]),
                    "labels": np.array([]),
                    "groups": np.array([]),
                    "features_pd": pd.DataFrame([]),
                    "labels_pd": pd.DataFrame([]),
                    "groups_pd": pd.DataFrame([]),
                },
            }
        }
    """
    ACSMobility = folktables.BasicProblem(
        features=[
            "AGEP",
            "SCHL",
            "MAR",
            "SEX",
            "DIS",
            "ESP",
            "CIT",
            "MIL",
            "ANC",
            "NATIVITY",
            "RELP",
            "DEAR",
            "DEYE",
            "DREM",
            "RAC1P",
            "GCL",
            "COW",
            "ESR",
            "WKHP",
            "JWMNP",
            "PINCP",
        ],
        target="MIG",
        target_transform=lambda x: x == 1,
        group="RAC1P",
        preprocess=lambda x: x.drop(x.loc[(x["AGEP"] <= 18) | (x["AGEP"] >= 35)].index),
        postprocess=lambda x: np.nan_to_num(x, -1),
    )

    data = create_dataset(year_list=year, state=state, ACSDataset=ACSMobility)
    return data


def create_public_coverage_dataset(year: list, state: list, group: str, target: str):
    """
    Downloads the US Census ACS data for the given year and state
    And creates the public coverage dataset

    If year is None, downloads all years
    If state is None, downloads all states
    If group is not specified, the default sensitive value will be "SEX"
    If target is not specified, the default target value will be "PINCP"

    Returns a dictionary in which the keys are the state names and the values are
    dictionaries where the keys are the years. For each year we will have a dictionary
    with the following keys:
        - features: numpy array of features
        - labels: numpy array of labels
        - groups: numpy array of groups
        - features_pd: pandas dataframe of features
        - labels_pd: pandas dataframe of labels
        - groups_pd: pandas dataframe of groups
    Example:
        {
            "AL": {
                "2014": {
                    "features": np.array([]),
                    "labels": np.array([]),
                    "groups": np.array([]),
                    "features_pd": pd.DataFrame([]),
                    "labels_pd": pd.DataFrame([]),
                    "groups_pd": pd.DataFrame([]),
                },
            }
        }
    """
    ACSPublicCoverage = folktables.BasicProblem(
        features=[
            "AGEP",
            "SCHL",
            "MAR",
            "SEX",
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
            "PINCP",
            "ESR",
            "ST",
            "FER",
            "RAC1P",
        ],
        target="PUBCOV",
        target_transform=lambda x: x == 1,
        group="SEX",
        preprocess=public_coverage_filter,
        postprocess=lambda x: np.nan_to_num(x, -1),
    )

    data = create_dataset(year_list=year, state=state, ACSDataset=ACSPublicCoverage)
    return data


def create_dataset(year_list, state, ACSDataset):
    global state_list
    global years
    data = {}
    years = years if year_list is None else year_list
    state_list = state_list if state is None else state
    for year in years:
        for state in state_list:
            data_source = ACSDataSource(survey_year=year, horizon="1-Year", survey="person")
            try:
                acs_data = data_source.get_data(states=[state], download=True)
                features, label, group = ACSDataset.df_to_numpy(acs_data)
                feature_pd, label_pd, group_pd = ACSDataset.df_to_pandas(acs_data)
                print("Downloaded state: ", state, " and year: ", year)
                if state not in data:
                    data[state] = {}

                if year not in data[state]:
                    data[state][year] = {}
                    data[state][year]["features"] = features
                    data[state][year]["labels"] = label
                    data[state][year]["groups"] = group
                    data[state][year]["features_pd"] = feature_pd
                    data[state][year]["labels_pd"] = label_pd
                    data[state][year]["groups_pd"] = group_pd
                else:
                    data[state][year]["features"] = np.concatenate((data[state]["features"], features), axis=0)
                    data[state][year]["labels"] = np.concatenate((data[state]["labels"], label), axis=0)
                    data[state][year]["groups"] = np.concatenate((data[state]["groups"], group), axis=0)
                    data[state][year]["features_pd"] = pd.concat((data[state]["features_pd"], feature_pd), axis=0)
                    data[state][year]["labels_pd"] = pd.concat((data[state]["labels_pd"], label_pd), axis=0)
                    data[state][year]["groups_pd"] = pd.concat((data[state]["groups_pd"], group_pd), axis=0)
            except:
                print("Error with state: ", state, " and year: ", year)
    return data


if __name__ == "__main__":
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

    assert args.dataset_name is not None, "Please specify a dataset name"

    if args.dataset_name == "income":
        data = create_income_dataset(year=year, state=state, group=group, target=target)
        dill.dump(data, open("../data/income_data.pkd", "wb"))
    elif args.dataset_name == "employment":
        data = create_employment_dataset(year=year, state=state, group=group, target=target)
        dill.dump(data, open("../data/employment_data.pkd", "wb"))
    elif args.dataset_name == "travel_time":
        data = create_travel_time_dataset(year=year, state=state, group=group, target=target)
        dill.dump(data, open("../data/travel_time_data.pkd", "wb"))
    elif args.dataset_name == "mobility":
        data = create_mobility_dataset(year=year, state=state, group=group, target=target)
        dill.dump(data, open("../data/mobility_data.pkd", "wb"))
    elif args.dataset_name == "public_coverage":
        data = create_public_coverage_dataset(year=year, state=state, group=group, target=target)
        dill.dump(data, open("../data/public_coverage_data.pkd", "wb"))
    elif args.dataset_name == "all":
        data = create_income_dataset(year=year, state=state, group=group, target=target)
        dill.dump(data, open("../data/income_data.pkd", "wb"))

        data = create_employment_dataset(year=year, state=state, group=group, target=target)
        dill.dump(data, open("../data/employment_data.pkd", "wb"))

        data = create_travel_time_dataset(year=year, state=state, group=group, target=target)
        dill.dump(data, open("../data/travel_time_data.pkd", "wb"))

        data = create_mobility_dataset(year=year, state=state, group=group, target=target)
        dill.dump(data, open("../data/mobility_data.pkd", "wb"))

        data = create_public_coverage_dataset(year=year, state=state, group=group, target=target)
        dill.dump(data, open("../data/public_coverage_data.pkd", "wb"))
    else:
        raise Exception("Invalid dataset name")
