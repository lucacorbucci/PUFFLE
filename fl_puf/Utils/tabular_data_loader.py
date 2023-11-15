import json
import os
from collections import Counter

import numpy as np
import pandas as pd
import seaborn as sns
import torch
from matplotlib.pyplot import figure
from scipy.io import arff
from sklearn.preprocessing import MinMaxScaler

from DPL.Datasets.dutch import TabularDataset
from fl_puf.Utils.utils import Utils

##############################################################################################################


def load_compas():
    def load_preproc_data_compas2(df2, protected_attributes=None):
        def custom_preprocessing(df2):
            """The custom pre-processing function is adapted from
            https://github.com/fair-preprocessing/nips2017/blob/master/compas/code/Generate_Compas_Data.ipynb
            """

            df2 = df2[
                [
                    "age",
                    "c_charge_degree",
                    "race",
                    "age_cat",
                    "score_text",
                    "sex",
                    "priors_count",
                    "days_b_screening_arrest",
                    "decile_score",
                    "is_recid",
                    "two_year_recid",
                    "c_jail_in",
                    "c_jail_out",
                    "compas_screening_date",
                ]
            ]

            date = df2["compas_screening_date"]

            # Indices of data samples to keep
            ix = df2["days_b_screening_arrest"] <= 30
            ix = (df2["days_b_screening_arrest"] >= -30) & ix
            ix = (df2["is_recid"] != -1) & ix
            ix = (df2["c_charge_degree"] != "O") & ix
            ix = (df2["score_text"] != "N/A") & ix
            ix = (df2["score_text"].notna()) & ix

            df2 = df2.loc[ix, :]
            df2["length_of_stay"] = (
                pd.to_datetime(df2["c_jail_out"]) - pd.to_datetime(df2["c_jail_in"])
            ).apply(lambda x: x.days)

            # Restrict races to African-American and Caucasian
            df2cut = df2.loc[
                ~df2["race"].isin(["Native American", "Hispanic", "Asian", "Other"]), :
            ]

            # Restrict the features to use
            df2cutQ = df2cut[
                [
                    "sex",
                    "race",
                    "age_cat",
                    "c_charge_degree",
                    "score_text",
                    "priors_count",
                    "is_recid",
                    "two_year_recid",
                    "length_of_stay",
                ]
            ].copy()

            # Quantize priors count between 0, 1-3, and >3
            def quantizePrior(x):
                if x <= 0:
                    return "0"
                elif 1 <= x <= 3:
                    return "1 to 3"
                else:
                    return "More than 3"

            # Quantize length of stay
            def quantizeLOS(x):
                if x <= 7:
                    return "<week"
                elif 7 < x <= 93:
                    return "week>&<3months"
                else:
                    return ">3 months"

            # fix age label
            def adjustAge(x):
                if x == "25 - 45":
                    return "25 to 45"
                else:
                    return x

            # Quantize score_text to MediumHigh
            def quantizeScore(x):
                if (x == "High") or (x == "Medium"):
                    return "MediumHigh"
                else:
                    return x

            def group_race(x):
                if x == "Caucasian":
                    return 1.0
                else:
                    return 0.0

            assert df2cutQ.isnull().sum().sum() == 0, "Encountered missing values!"

            df2cutQ["priors_count"] = df2cutQ["priors_count"].apply(
                lambda x: quantizePrior(x)
            )
            df2cutQ["length_of_stay"] = df2cutQ["length_of_stay"].apply(
                lambda x: quantizeLOS(x)
            )
            df2cutQ["score_text"] = df2cutQ["score_text"].apply(
                lambda x: quantizeScore(x)
            )
            df2cutQ["age_cat"] = df2cutQ["age_cat"].apply(lambda x: adjustAge(x))
            # check age categories after adjusting
            # print('unique vals after quantizing')
            # print(df2cutQ['age_cat'].unique())
            # print(df2cutQ['score_text'].unique())
            # print(df2cutQ['priors_count'].unique())
            # print(df2cutQ['length_of_stay'].unique())

            # Recode sex and race
            df2cutQ["sex"] = df2cutQ["sex"].replace({"Female": 0.0, "Male": 1.0})
            df2cutQ["race"] = df2cutQ["race"].apply(lambda x: group_race(x))

            features = [
                "two_year_recid",
                "sex",
                "race",
                "age_cat",
                "priors_count",
                "c_charge_degree",
                "length_of_stay",
            ]

            # Pass vallue to df2
            df2 = df2cutQ[features]

            df2["compas_screening_date"] = dates
            return df2

        XD_features = [
            "age_cat",
            "c_charge_degree",
            "priors_count",
            "sex",
            "race",
            "compas_screening_date",
            "length_of_stay",
        ]
        D_features = (
            ["sex", "race"] if protected_attributes is None else protected_attributes
        )
        Y_features = ["two_year_recid"]
        X_features = list(set(XD_features) - set(D_features))
        categorical_features = [
            "age_cat",
            "c_charge_degree",
            "priors_count",
            "length_of_stay",
            "compas_screening_date",
        ]

        # privileged classes
        all_privileged_classes = {"sex": [1.0], "race": [1.0]}

        # protected attribute maps
        all_protected_attribute_maps = {
            "sex": {1.0: "Male", 0.0: "Female"},
            "race": {1.0: "Caucasian", 0.0: "Not Caucasian"},
        }

        return custom_preprocessing(df2)

    # full recidivism dataset:
    # _df2 = pd.read_csv("https://raw.githubusercontent.com/propublica/compas-analysis/master/compas-scores-two-years.csv")
    # often only violent recidivism is considered:
    # _df2 = pd.read_csv("https://raw.githubusercontent.com/propublica/compas-analysis/master/compas-scores-two-years-violent.csv")
    _df2 = pd.read_csv("data/Tabular/compas/compas-scores-two-years-violent.csv")
    df_compas = load_preproc_data_compas2(_df2, ["sex"])

    df_compas["year"] = pd.DatetimeIndex(df_compas["compas_screening_date"]).year
    df_compas.head()

    feature_columns_compas = [
        "sex",
        "race",
        "age_cat",
        "priors_count",
        "c_charge_degree",
        "year",
        "length_of_stay",
    ]

    metadata_compas = {
        "name": "Compas",
        "protected_atts": [
            "race",
            "sex",
        ],  # when using less, these are chosen in order from 0th
        "code": ["CO1", "CO2"],
        "protected_att_values": [0, 0],
        "protected_att_descriptions": ["Race \\neq White", "Gender = Female"],
        "target_variable": "two_year_recid",
    }

    return df_compas, feature_columns_compas, metadata_compas


def load_adult(dataset_path):
    adult_feat_cols = [
        "age",
        "workclass",
        "fnlwgt",
        "education",
        "education-num",
        "marital-status",
        "occupation",
        "relationship",
        "race",
        "sex",
        "capital-gain",
        "capital-loss",
        "hours-per-week",
        "native-country",
        "sex_binary",
        "race_binary",
        "age_binary",
    ]

    adult_columns_names = (
        "age",
        "workclass",  # Private, Self-emp-not-inc, Self-emp-inc, Federal-gov, Local-gov, State-gov, Without-pay, Never-worked.
        "fnlwgt",  # "weight" of that person in the dataset (i.e. how many people does that person represent) -> https://www.kansascityfed.org/research/datamuseum/cps/coreinfo/keyconcepts/weights
        "education",
        "education-num",
        "marital-status",
        "occupation",
        "relationship",
        "race",
        "sex",
        "capital-gain",
        "capital-loss",
        "hours-per-week",
        "native-country",
        "income",
    )

    df_adult = pd.read_csv(dataset_path + "adult.data", names=adult_columns_names)
    df_adult["sex_binary"] = np.where(df_adult["sex"] == " Male", 1, 0)
    df_adult["race_binary"] = np.where(df_adult["race"] == " White", 1, 0)
    df_adult["age_binary"] = np.where(
        (df_adult["age"] > 25) & (df_adult["age"] < 60), 1, 0
    )

    y = np.zeros(len(df_adult))

    y[df_adult["income"] == " >50K"] = 1
    df_adult["income_binary"] = y
    del df_adult["income"]

    metadata_adult = {
        "name": "Adult",
        "protected_atts": ["sex_binary", "race_binary", "age_binary"],
        "protected_att_values": [0, 0, 0],
        "code": ["AD1", "AD2", "AD3"],
        "protected_att_descriptions": [
            "Gender = Female",
            "Race \\neq white",
            "Age <25 | >60",
        ],
        "target_variable": "income_binary",
    }

    return df_adult, adult_feat_cols, metadata_adult


def load_german():
    df_german = pd.read_csv("data/Tabular/german_credit/german_credit_data.csv")

    df_german["age_binary"] = np.where(df_german["Age (years)"] > 25, 1, 0)
    del df_german["Age (years)"]

    german_feature_columns = df_german.columns[1:]

    metadata_german = {
        "name": "German",
        "protected_atts": ["age_binary"],
        "protected_att_values": [0],
        "code": ["GE1"],
        "protected_att_descriptions": ["Age > 25"],
        "target_variable": "Creditability",
    }

    return df_german, german_feature_columns, metadata_german


def load_kdd():
    kdd_columns = [
        "age",
        "class of worker",
        "industry code",
        "occupation code",
        "adjusted gross income",
        "education",
        "wage per hour",
        "enrolled in edu inst last wk",
        "marital status",
        "major industry code",
        "major occupation code",
        "mace",
        "hispanic Origin",
        "sex",
        "member of a labor union",
        "reason for unemployment",
        "full or part time employment stat",
        "capital gains",
        "capital losses",
        "divdends from stocks",
        "federal income tax liability",
        "tax filer status",
        "region of previous residence",
        "state of previous residence",
        "detailed household and family stat",
        "detailed household summary in household",
        "instance weight",
        "migration code-change in msa",
        "migration code-change in reg",
        "migration code-move within reg",
        "live in this house 1 year ago",
        "migration prev res in sunbelt",
        "num persons worked for employer",
        "family members under 18",
        "total person earnings",
        "country of birth father",
        "country of birth mother",
        "country of birth self",
        "citizenship",
        "total person income",
        "own business or self employed",
        "taxable income amount",
        "fill inc questionnaire for veteran's admin",
        "veterans benefits",
        "weeks worked in year",
    ]

    feature_columns_kdd = [
        "age",
        "class of worker",
        "industry code",
        "occupation code",
        "adjusted gross income",
        "education",
        "wage per hour",
        "enrolled in edu inst last wk",
        "marital status",
        "major industry code",
        "major occupation code",
        "mace",
        "hispanic Origin",
        "member of a labor union",
        "reason for unemployment",
        "full or part time employment stat",
        "capital gains",
        "capital losses",
        "divdends from stocks",
        "federal income tax liability",
        "tax filer status",
        "region of previous residence",
        "state of previous residence",
        "detailed household and family stat",
        "detailed household summary in household",
        "instance weight",
        "migration code-change in msa",
        "migration code-change in reg",
        "migration code-move within reg",
        "live in this house 1 year ago",
        "migration prev res in sunbelt",
        "num persons worked for employer",
        "family members under 18",
        "total person earnings",
        "country of birth father",
        "country of birth mother",
        "country of birth self",
        "citizenship",
        "own business or self employed",
        "fill inc questionnaire for veteran's admin",
        "veterans benefits",
        "weeks worked in year",
    ]

    kdd_census_df = pd.read_csv(
        "data/Tabular/kdd/KDD-census-income.data", names=kdd_columns
    )

    kdd_census_df = kdd_census_df[kdd_census_df["sex"] != " Not in universe"]
    kdd_census_df["sex_binary"] = np.where(kdd_census_df["sex"] == " Yes", 1, 0)
    del kdd_census_df["sex"]
    y = np.zeros(len(kdd_census_df))
    y[kdd_census_df["taxable income amount"] == " 50000+."] = 1
    kdd_census_df["income_binary"] = y
    del kdd_census_df["taxable income amount"]

    # print(kdd_census_df.head())
    # NOTE: e.g. fairness data survey recommends dropping some columns due to high missingness, e.g.:
    print(kdd_census_df["migration code-move within reg"].unique())

    # drop missing values
    # kdd_census_df.dropna(inplace=True, axis=1)
    # print(kdd_census_df.isnull().sum())
    assert kdd_census_df.isnull().sum().sum() == 0, "Encountered missing values!"

    metadata_kdd = {
        "name": "KDD-Census-Income",
        "protected_atts": ["sex_binary"],
        "protected_att_values": [1],
        "code": ["CE1"],
        "protected_att_descriptions": ["Gender = Female"],
        "target_variable": "income_binary",
    }
    return kdd_census_df, feature_columns_kdd, metadata_kdd


def load_bank_marketing():
    bank_marketing_df = pd.read_csv(
        "data/Tabular/bank_marketing/bank-full.csv", sep=";"
    )
    # count number of 'married' in marital column
    print(bank_marketing_df["marital"].value_counts())

    bank_marketing_df["marital_binary"] = np.where(
        bank_marketing_df["marital"] == "married", 1, 0
    )
    bank_marketing_df["age_binary"] = np.where(
        (bank_marketing_df["age"] > 25) & (bank_marketing_df["age"] < 60), 1, 0
    )

    bank_marketing_df["y_binary"] = np.where(bank_marketing_df["y"] == "no", 0, 1)
    # print(bank_marketing_df.head())

    del bank_marketing_df["y"]
    del bank_marketing_df["age"]
    del bank_marketing_df["marital"]

    feature_columns_bank = bank_marketing_df.columns[:-1]

    # check that no missing vals
    assert bank_marketing_df.isnull().sum().sum() == 0, "Encountered missing values!"

    metadata_bank_marketing = {
        "name": "Bank marketing",
        "protected_atts": ["marital_binary", "age_binary"],
        "protected_att_values": [0, 0],
        "code": ["BM1", "BM2"],
        "protected_att_descriptions": ["Marital status \\neq married", "Age <25 | >60"],
        "target_variable": "y_binary",
    }

    return bank_marketing_df, feature_columns_bank, metadata_bank_marketing


def load_credit_card():
    credit_card_df = pd.read_csv("data/Tabular/credit_card/UCI_Credit_Card.csv")
    credit_card_df["marital_binary"] = np.where(credit_card_df["MARRIAGE"] == 1, 1, 0)
    credit_card_df["education_binary"] = np.where(
        credit_card_df["EDUCATION"] == 2, 1, 0
    )
    credit_card_df["sex_binary"] = np.where(credit_card_df["SEX"] == 2, 1, 0)
    # print(credit_card_df['sex_binary'].mean())
    # print(credit_card_df['marital_binary'].mean())
    # print(credit_card_df['education_binary'].mean())

    del credit_card_df["MARRIAGE"]
    del credit_card_df["EDUCATION"]
    del credit_card_df["SEX"]
    del credit_card_df["ID"]  # drop unique id

    assert credit_card_df.isnull().sum().sum() == 0, "Encountered missing values!"

    credit_feature_columns = [
        "LIMIT_BAL",
        "AGE",
        "PAY_0",
        "PAY_2",
        "PAY_3",
        "PAY_4",
        "PAY_5",
        "PAY_6",
        "BILL_AMT1",
        "BILL_AMT2",
        "BILL_AMT3",
        "BILL_AMT4",
        "BILL_AMT5",
        "BILL_AMT6",
        "PAY_AMT1",
        "PAY_AMT2",
        "PAY_AMT3",
        "PAY_AMT4",
        "PAY_AMT5",
        "PAY_AMT6",
        "marital_binary",
        "education_binary",
        "sex_binary",
    ]
    metadata_credit_card = {
        "name": "Credit card clients",
        "protected_atts": ["sex_binary", "marital_binary", "education_binary"],
        "protected_att_values": [1, 0, 0],
        "code": ["CC1", "CC2", "CC3"],
        "protected_att_descriptions": [
            "Gender = Female",
            "Marital status \\neq married",
            "Education \\neq University",
        ],
        "target_variable": "default.payment.next.month",
    }
    return credit_card_df, credit_feature_columns, metadata_credit_card


def load_communities_and_crime():
    communities_and_crime_df = pd.read_csv(
        "data/Tabular/crimedata.csv", encoding="latin"
    )
    communities_and_crime_df = communities_and_crime_df.dropna(axis=1, how="all")

    communities_and_crime_df = communities_and_crime_df[
        communities_and_crime_df["ViolentCrimesPerPop"] != "?"
    ]
    communities_and_crime_df = communities_and_crime_df[
        communities_and_crime_df["ViolentCrimesPerPop"] != "y"
    ]

    communities_and_crime_df["ViolentCrimesPerPop"] = communities_and_crime_df[
        "ViolentCrimesPerPop"
    ].astype(float)
    communities_and_crime_df["is_violent"] = np.where(
        communities_and_crime_df["ViolentCrimesPerPop"] > 0.7, 1, 0
    )
    communities_and_crime_df["is_black"] = np.where(
        communities_and_crime_df["racepctblack"] > 0.06, 1, 0
    )

    race_black = np.zeros(len(communities_and_crime_df))

    race_black[communities_and_crime_df["racepctblack"] > 0.06] = 1
    communities_and_crime_df["race_black"] = race_black

    y = np.zeros(len(communities_and_crime_df))

    a = pd.to_numeric(communities_and_crime_df["ViolentCrimesPerPop"])
    y[a > a.median()] = 1

    communities_and_crime_df["y_col"] = y

    del communities_and_crime_df["racepctblack"]
    del communities_and_crime_df["ViolentCrimesPerPop"]

    feature_columns_comm = communities_and_crime_df.columns[:-1]

    # drop missing values
    # kdd_census_df.dropna(inplace=True, axis=1)
    # print(kdd_census_df.isnull().sum())
    assert (
        communities_and_crime_df.isnull().sum().sum() == 0
    ), "Encountered missing values!"

    metadata_communities = {
        "name": "Communities and crime",
        "code": ["CM1"],
        "protected_atts": ["is_black"],
        "protected_att_values": [1],
        "protected_att_descriptions": ["Race = Black"],
        "target_variable": "is_violent",
    }

    return communities_and_crime_df, feature_columns_comm, metadata_communities


def load_diabetes():
    diabetes_df = pd.read_csv("data/Tabular/diabetes/diabetic_data.csv")

    diabetes_df = diabetes_df.drop(
        diabetes_df[diabetes_df["gender"] == "Unknown/Invalid"].index
    )

    diabetes_df["gender_binary"] = np.where(diabetes_df["gender"] == "Male", 1, 0)
    diabetes_df["readmitted_binary"] = np.where(diabetes_df["readmitted"] == "NO", 0, 1)
    diabetes_df = diabetes_df.dropna(
        subset=["race", "diag_1", "diag_2", "diag_3"], how="all"
    )
    diabetes_df = diabetes_df.drop(diabetes_df[diabetes_df["race"] == "?"].index)
    diabetes_df = diabetes_df.drop(diabetes_df[diabetes_df["diag_1"] == "?"].index)
    diabetes_df = diabetes_df.drop(diabetes_df[diabetes_df["diag_2"] == "?"].index)
    diabetes_df = diabetes_df.drop(diabetes_df[diabetes_df["diag_3"] == "?"].index)

    # print(diabetes_df['gender'].unique())
    # print(diabetes_df['readmitted'].unique())
    # print(diabetes_df['race'].unique())
    # print(diabetes_df['diag_1'].unique())
    # print(diabetes_df['diag_2'].unique())
    # print(diabetes_df['diag_3'].unique())
    # replace some values encoded as nans with 'none' as in UCI description, i.e., it wasn't measured, not just missing
    diabetes_df["A1Cresult"].fillna("none", inplace=True)
    diabetes_df["max_glu_serum"].fillna("none", inplace=True)
    # print(diabetes_df['max_glu_serum'].unique())
    # print(diabetes_df['A1Cresult'].unique())

    del diabetes_df["readmitted"]
    del diabetes_df["gender"]
    # drop some features as per fairness data survey
    del diabetes_df["encounter_id"]
    del diabetes_df["patient_nbr"]
    del diabetes_df["weight"]
    del diabetes_df["payer_code"]
    del diabetes_df["medical_specialty"]

    feature_columns_diabetes = diabetes_df.columns[:-1]
    # print( [ diabetes_df.isin({tmp:['?']}).any() for tmp in diabetes_df.columns ] )

    # print(diabetes_df.head())

    # drop missing values
    # diabetes_df.dropna(inplace=True, axis=1)
    # print(diabetes_df.isnull().sum())
    assert diabetes_df.isnull().sum().sum() == 0, "Encountered missing values!"

    metadata_diabetes = {
        "name": "Diabetes",
        "protected_atts": ["gender_binary"],
        "code": ["DI1"],
        "protected_att_values": [0],
        "protected_att_descriptions": ["Gender = Female"],
        "target_variable": "readmitted_binary",
        "dummy_cols": [
            "age",
            "race",
            "admission_type_id",
            "discharge_disposition_id",
            "admission_source_id",
            "diag_1",
            "diag_2",
            "diag_3",
            "max_glu_serum",
            "A1Cresult",
            "metformin",
            "repaglinide",
            "nateglinide",
            "chlorpropamide",
            "glimepiride",
            "acetohexamide",
            "glipizide",
            "glyburide",
            "tolbutamide",
            "pioglitazone",
            "rosiglitazone",
            "acarbose",
            "miglitol",
            "troglitazone",
            "tolazamide",
            "examide",
            "citoglipton",
            "insulin",
            "glyburide-metformin",
            "glipizide-metformin",
            "glimepiride-pioglitazone",
            "metformin-rosiglitazone",
            "metformin-pioglitazone",
            "change",
            "diabetesMed",
        ],  # columns that can be 1-hot encoded
    }

    return diabetes_df, feature_columns_diabetes, metadata_diabetes


def load_ricci():
    ricci_df = pd.read_csv("data/Tabular/ricci/Ricci.csv")
    ricci_race_binary = np.ones(len(ricci_df))

    ricci_race_binary[ricci_df["Race"] != "W"] = 0

    ricci_df["race_binary"] = ricci_race_binary

    del ricci_df["Race"]

    feature_cols_ricci = ["Position", "Oral", "Written", "Combine", "race_binary"]

    # drop missing values
    # kdd_census_df.dropna(inplace=True, axis=1)
    # print(kdd_census_df.isnull().sum())
    assert ricci_df.isnull().sum().sum() == 0, "Encountered missing values!"

    metadata_ricci = {
        "name": "Ricci",
        "protected_atts": ["race_binary"],
        "code": ["RI1"],
        "protected_att_values": [0],
        "protected_att_descriptions": ["Race \\neq White"],
        "target_variable": "Class",
    }

    return ricci_df, feature_cols_ricci, metadata_ricci


def load_oulad():
    oulad_df = pd.read_csv("data/Tabular/oulad/studentInfo.csv")
    oulad_df["gender_binary"] = np.where(oulad_df["gender"] == "F", 1, 0)

    oulad_df = oulad_df[oulad_df["final_result"] != "Withdrawn"]

    oulad_df["final_result_binary"] = np.where(oulad_df["final_result"] == "Pass", 1, 0)

    del oulad_df["gender"]
    del oulad_df["final_result"]

    # drop missing values
    # kdd_census_df.dropna(inplace=True, axis=1)
    # print(kdd_census_df.isnull().sum())
    assert oulad_df.isnull().sum().sum() == 0, "Encountered missing values!"

    feature_columns_oulad = [
        "code_module",
        "code_presentation",
        "id_student",
        "region",
        "highest_education",
        "imd_band",
        "age_band",
        "num_of_prev_attempts",
        "studied_credits",
        "disability",
        "gender_binary",
    ]

    metadata_oulad = {
        "name": "OULAD dataset",
        "protected_atts": ["gender_binary"],
        "code": ["OU1"],
        "protected_att_values": [0],
        "protected_att_descriptions": ["Gender = Female"],
        "target_variable": "final_result_binary",
    }

    return oulad_df, feature_columns_oulad, metadata_oulad


def load_maths_student():
    st_math_df = pd.read_csv("data/Tabular/math_student/student-mat.csv", sep=";")

    st_math_df["sex_binary"] = np.where(st_math_df["sex"] == "F", 0, 1)
    st_math_df["age_binary"] = np.where(st_math_df["age"] < 18, 0, 1)
    st_math_df["y_binary"] = np.where(st_math_df["G3"] >= 10, 1, 0)

    del st_math_df["G3"]
    del st_math_df["sex"]
    del st_math_df["age"]

    # drop missing values
    # kdd_census_df.dropna(inplace=True, axis=1)
    # print(kdd_census_df.isnull().sum())
    assert st_math_df.isnull().sum().sum() == 0, "Encountered missing values!"

    metadata_math = {
        "name": "Student mathematics",
        "code": ["ST1", "ST2"],
        "protected_atts": ["sex_binary", "age_binary"],
        "protected_att_values": [0, 0],
        "protected_att_descriptions": ["Gender = Female", "Age < 18"],
        "target_variable": "y_binary",
    }
    return st_math_df, st_math_df.columns[:-1], metadata_math


def load_portuguese_student():
    st_por_df = pd.read_csv("data/Tabular/portuguese_student/student-por.csv", sep=";")
    st_por_df["sex_binary"] = np.where(st_por_df["sex"] == "F", 0, 1)
    st_por_df["age_binary"] = np.where(st_por_df["age"] < 18, 0, 1)
    st_por_df["y_binary"] = np.where(st_por_df["G3"] >= 10, 1, 0)

    del st_por_df["G3"]
    del st_por_df["sex"]
    del st_por_df["age"]

    # drop missing values
    # kdd_census_df.dropna(inplace=True, axis=1)
    # print(kdd_census_df.isnull().sum())
    assert st_por_df.isnull().sum().sum() == 0, "Encountered missing values!"

    metadata_por = {
        "name": "Student portuguese",
        "code": ["ST3", "ST4"],
        "protected_atts": ["sex_binary", "age_binary"],
        "protected_att_values": [0, 0],
        "protected_att_descriptions": ["Gender = Female", "Age < 18"],
        "target_variable": "y_binary",
    }
    return st_por_df, st_por_df.columns[:-1], metadata_por


def load_lsac():
    df_lsac = pd.read_stata("data/Tabular/law/lawschs1_1.dta").dropna()
    df_lsac = df_lsac.drop(columns=["race"])

    feature_columns_lsac = df_lsac.columns[:-1]

    # drop missing values
    # kdd_census_df.dropna(inplace=True, axis=1)
    # print(kdd_census_df.isnull().sum())
    assert df_lsac.isnull().sum().sum() == 0, "Encountered missing values!"

    metadata_lsac = {
        "name": "LSAC",
        "protected_atts": ["gender", "white"],
        "code": ["LS1", "LS2"],
        "protected_att_values": [0, 0],
        "protected_att_descriptions": ["Gender = Female", "Race \\neq White"],
        "target_variable": "enroll",
    }

    return df_lsac, feature_columns_lsac, metadata_lsac


def plot_distribution(distributions, title):
    counters = {}
    key_list = set()
    nodes_data = []
    for distribution in distributions:
        counter = Counter(distribution)
        nodes_data.append(counter)
        for key, value in counter.items():
            key_list.add(key)

    for node_data in nodes_data:
        for key in key_list:
            if key not in node_data:
                if key not in counters:
                    counters[key] = []
                counters[key].append(0)
            else:
                if key not in counters:
                    counters[key] = []
                counters[key].append(node_data[key])

    figure(figsize=(25, 8), dpi=80)
    indexes = np.arange(len(distributions))

    legend_values = []
    colors = ["blue", "green", "red", "purple", "orange", "yellow", "pink", "brown"]
    markers = ["+", "o", "x", "*", "v", "^", "<", ">"]
    for i, (key, value) in enumerate(counters.items()):
        plt.scatter(
            indexes, value, marker=markers[i], linewidths=4, s=100, color=colors[i]
        )
        legend_values.append(key)
    plt.xticks(indexes, list(range(0, len(indexes))))
    plt.rcParams.update({"font.size": 16})
    plt.xlabel("Nodes")
    plt.ylabel("Samples")
    plt.xticks(rotation=90)
    plt.title(title)
    plt.legend(legend_values)
    plt.savefig("distribution.png")

    # # Plots the final disparity of the baseline, fixed and tunable models for each node
    # baseline_disparity_dict = {}
    # fixed_disparity_dict = {}
    # tunable_disparit_dict = {}

    # for node_name, column in zip(nodes, baseline.columns):
    #     # get the last value of the column
    #     if column in baseline.columns:
    #         baseline_disparity_dict[node_name] = baseline[column].iloc[-1]
    #     # check if the dataframe fixed and tunable are None or not
    #     # and if the column is in the dataframe
    #     if column in fixed.columns:
    #         fixed_disparity_dict[node_name] = fixed[column].iloc[-1]
    #     if column in tunable.columns:
    #         tunable_disparit_dict[node_name] = tunable[column].iloc[-1]

    # baseline_disparity = []
    # fixed_disparity = []
    # tunable_disparity = []
    # for node in sorted_nodes:
    #     if node in baseline_disparity_dict:
    #         baseline_disparity.append(baseline_disparity_dict[node])
    #     if fixed_disparity_dict != {} and node in fixed_disparity_dict:
    #         fixed_disparity.append(fixed_disparity_dict[node])
    #     if tunable_disparit_dict != {} and node in tunable_disparit_dict:
    #         tunable_disparity.append(tunable_disparit_dict[node])

    # # get the training disparity
    # nodes_with_disparity = {}
    # for node_name, column in zip(nodes, disparity_dataset.columns):
    #     # get the last value of the column
    #     nodes_with_disparity[node_name] = disparity_dataset[column].iloc[-1]

    # training_disparity = [nodes_with_disparity[node_name] for node_name in sorted_nodes]

    # width = 0.27
    # figure(figsize=(25, 8), dpi=80)
    # indexes = np.arange(len(sorted_nodes))

    # plt.scatter(
    #     indexes, baseline_disparity, marker="x", linewidths=4, s=100, color="red"
    # )
    # if fixed_disparity:
    #     plt.scatter(
    #         indexes, fixed_disparity, marker="o", linewidths=4, s=100, color="green"
    #     )
    # if tunable_disparity:
    #     plt.scatter(
    #         indexes, tunable_disparity, marker="+", linewidths=4, s=100, color="blue"
    #     )
    # plt.scatter(
    #     indexes, training_disparity, marker="*", linewidths=4, s=100, color="black"
    # )

    # plt.plot(indexes, baseline_disparity, linewidth=1, color="red")
    # if fixed_disparity:
    #     plt.plot(indexes, fixed_disparity, linewidth=1, color="green")
    # if tunable_disparity:
    #     plt.plot(indexes, tunable_disparity, linewidth=1, color="blue")
    # plt.plot(indexes, training_disparity, linewidth=1, color="black")

    # plt.axhline(y=target, color="purple", linestyle="--")

    # if tunable_disparity and fixed_disparity:
    #     plt.legend(["Baseline", "Fixed", "Tunable", "Dataset Disparity"])
    # else:
    #     plt.legend(["Baseline", "Dataset Disparity"])
    # plt.xticks(indexes, sorted_nodes)
    # plt.rcParams.update({"font.size": 16})
    # plt.xlabel("Nodes")
    # plt.ylabel("Disparity")
    # plt.xticks(rotation=90)
    # plt.title(title)
    # plt.savefig(file_name)


#########################################################################################


def get_tabular_data(
    num_clients: int,
    do_iid_split: bool,
    groups_balance_factor: float,
    priv_balance_factor: float,
    dataset_name: str,
    num_sensitive_features: int,
    approach: str,
    num_nodes: int,
    ratio_unfair_nodes: float,
    opposite_direction: bool,
    ratio_unfairness: tuple,
    dataset_path=None,
    group_to_reduce: tuple = None,
    group_to_increment: tuple = None,
    number_of_samples_per_node: int = None,
    opposite_group_to_reduce: tuple = None,
    opposite_group_to_increment: tuple = None,
    opposite_ratio_unfairness: tuple = None,
):
    X, z, y = get_tabular_numpy_dataset(
        dataset_name=dataset_name,
        num_sensitive_features=num_sensitive_features,
        dataset_path=dataset_path,
    )
    z = z[:, 0]
    print(f"Data shapes: x={X.shape}, y={y.shape}, z={z.shape}")
    # Prepare training data held by each client
    # Metadata is a list with 0 if the client is fair, 0 otherwise
    client_data, metadata = generate_clients_biased_data_mod(
        X=X,
        y=y,
        z=z,
        approach=approach,
        num_nodes=num_nodes,
        ratio_unfair_nodes=ratio_unfair_nodes,
        opposite_direction=opposite_direction,
        ratio_unfairness=ratio_unfairness,
        group_to_reduce=group_to_reduce,
        group_to_increment=group_to_increment,
        number_of_samples_per_node=number_of_samples_per_node,
        opposite_group_to_reduce=opposite_group_to_reduce,
        opposite_group_to_increment=opposite_group_to_increment,
        opposite_ratio_unfairness=opposite_ratio_unfairness,
    )
    disparities = Utils.compute_disparities_debug(client_data)
    plot_bar_plot(
        title=f"{approach}",
        disparities=disparities,
        nodes=[f"{i}" for i in range(len(client_data))],
    )

    education_level = []
    for client in client_data:
        education_level.append([sample["x"][6] for sample in client])

    plot_distribution(education_level, title="Education Level")

    # plot_bar_plot(
    #     title=f"Education Level",
    #     disparities=[client["x"][6] for client in client_data],
    #     nodes=[f"{i}" for i in range(len(client_data))],
    # )

    return client_data, disparities, metadata  # , N_is, props_positive


def egalitarian_approach(X, y, z, num_nodes, number_of_samples_per_node=None):
    """
    With this approach we want to distribute the data among the nodes in an egalitarian way.
    This means that each node has the same amount of data and the same ratio of each group

    params:
    X: numpy array of shape (N, D) where N is the number of samples and D is the number of features
    y: numpy array of shape (N, ) where N is the number of samples. Here we have the samples labels
    z: numpy array of shape (N, ) where N is the number of samples. Here we have the samples sensitive features
    num_nodes: number of nodes to generate
    number_of_samples_per_node: number of samples that we want in each node. Can be None, in this case we just use
        len(y)//num_nodes
    """
    combinations = [(target, sensitive_value) for target, sensitive_value in zip(y, z)]
    possible_combinations = set(combinations)
    data = {}
    for combination, x_, y_, z_ in zip(combinations, X, y, z):
        if combination not in data:
            data[combination] = []
        data[combination].append({"x": x_, "y": y_, "z": z_})

    samples_from_each_group = min(list(Counter(combinations).values())) // num_nodes

    if number_of_samples_per_node:
        assert (
            samples_from_each_group * len(possible_combinations)
            >= number_of_samples_per_node
        ), "Too many samples per node, choose a different number of samples per node"
        if (
            samples_from_each_group * len(possible_combinations)
            >= number_of_samples_per_node
        ):
            to_be_removed = (
                samples_from_each_group * len(possible_combinations)
                - number_of_samples_per_node
            ) // len(possible_combinations)
            samples_from_each_group -= to_be_removed

    # create the nodes
    nodes = []
    for i in range(num_nodes):
        nodes.append([])
        # fill the nodes
        for combination in data:
            nodes[i].extend(data[combination][:samples_from_each_group])
            data[combination] = data[combination][samples_from_each_group:]

    return nodes, data


def create_unfair_nodes(
    nodes_to_unfair: list,
    remaining_data: dict,
    group_to_reduce: tuple,
    group_to_increment: tuple,
    ratio_unfairness: tuple,
):
    """
    This function creates the unfair nodes. It takes the nodes that we want to be unfair and the remaining data
    and it returns the unfair nodes created by reducing the group_to_reduce and incrementing the group_to_increment
    based on the ratio_unfairness

    params:
    nodes_to_unfair: list of nodes that we want to make unfair
    remaining_data: dictionary with the remaining data that we will use to replace the
        samples that we remove from the nodes_to_unfair
    group_to_reduce: the group that we want to be unfair. For instance, in the case of binary target and binary sensitive value
        we could have (0,0), (0,1), (1,0) or (1,1)
    group_to_increment: the group that we want to increment. For instance, in the case of binary target and binary sensitive value
        we could have (0,0), (0,1), (1,0) or (1,1)
    ratio_unfairness: tuple (min, max) where min is the minimum ratio of samples that we want to remove from the group_to_reduce
    """
    assert (
        remaining_data[group_to_reduce] != []
    ), "Choose a different group to be unfair"
    # remove the samples from the group that we want to be unfair
    unfair_nodes = []
    number_of_samples_to_add = []
    for node in nodes_to_unfair:
        node_data = []
        count_sensitive_group_samples = 0
        for sample in node:
            if (sample["y"], sample["z"]) == group_to_reduce:
                count_sensitive_group_samples += 1

        current_ratio = np.random.uniform(ratio_unfairness[0], ratio_unfairness[1])
        samples_to_be_removed = int(count_sensitive_group_samples * current_ratio)
        number_of_samples_to_add.append(samples_to_be_removed)

        for sample in node:
            if (
                sample["y"],
                sample["z"],
            ) == group_to_reduce and samples_to_be_removed > 0:
                samples_to_be_removed -= 1
            else:
                node_data.append(sample)
        unfair_nodes.append(node_data)

    assert sum(number_of_samples_to_add) < len(
        remaining_data[group_to_increment]
    ), "Choose a different group to increment or reduce the ratio_unfairness"
    # now we have to add the same amount of data taken from group_to_unfair
    for node, samples_to_add in zip(unfair_nodes, number_of_samples_to_add):
        node.extend(remaining_data[group_to_increment][:samples_to_add])
        remaining_data[group_to_increment] = remaining_data[group_to_increment][
            samples_to_add:
        ]

    return unfair_nodes


def representative_diversity_approach(X, y, z, num_nodes, number_of_samples_per_node):
    """
    With this approach we want to distribute the data among the nodes in a representative diversity way.
    This means that each node has the same ratio of each group that we are observing in the dataset

    params:
    X: numpy array of shape (N, D) where N is the number of samples and D is the number of features
    y: numpy array of shape (N, ) where N is the number of samples. Here we have the samples labels
    z: numpy array of shape (N, ) where N is the number of samples. Here we have the samples sensitive features
    num_nodes: number of nodes to generate
    number_of_samples_per_node: number of samples that we want in each node. Can be None, in this case we just use
        len(y)//num_nodes
    """
    samples_per_node = (
        number_of_samples_per_node
        if number_of_samples_per_node
        else len(y) // num_nodes
    )
    # create the nodes sampling from the dataset wihout replacement
    dataset = [{"x": x_, "y": y_, "z": z_} for x_, y_, z_ in zip(X, y, z)]
    # shuffle the dataset
    np.random.shuffle(dataset)

    # Distribute the data among the nodes with a random sample from the dataset
    # considering the number of samples per node
    nodes = []
    for i in range(num_nodes):
        nodes.append([])
        nodes[i].extend(dataset[:samples_per_node])
        dataset = dataset[samples_per_node:]

    # Create the dictionary with the remaining data
    remaining_data = {}
    for sample in dataset:
        if (sample["y"], sample["z"]) not in remaining_data:
            remaining_data[(sample["y"], sample["z"])] = []
        remaining_data[(sample["y"], sample["z"])].append(sample)

    return nodes, remaining_data


def generate_clients_biased_data_mod(
    X,
    y,
    z,
    approach: str,
    num_nodes: int,
    ratio_unfair_nodes: float,
    opposite_direction: bool,
    ratio_unfairness: tuple,
    group_to_reduce: tuple = None,
    group_to_increment: tuple = None,
    number_of_samples_per_node: int = None,
    opposite_group_to_reduce: tuple = None,
    opposite_group_to_increment: tuple = None,
    opposite_ratio_unfairness: tuple = None,
):
    """
    This function generates the data for the clients.

    params:
    X: numpy array of shape (N, D) where N is the number of samples and D is the number of features
    y: numpy array of shape (N, ) where N is the number of samples. Here we have the samples labels
    z: numpy array of shape (N, ) where N is the number of samples. Here we have the samples sensitive features
    num_nodes: number of nodes to generate
    approach: type of approach we want to use to distribute the data among the fair clients. This can be egalitarian or representative
    ratio_unfair_nodes: the fraction of unfair clients we want to have in the experiment
    opposite_direction: true if we want to allow different nodes to have different majoritiarian classes. For instance,
        we could have some nodes with a max disparity that depends on the majority class being 0 and other nodes with a max disparity
        that depends on the majority class being 1.
    group_to_reduce: the group that we want to be unfair. For instance, in the case of binary target and binary sensitive value
        we could have (0,0), (0,1), (1,0) or (1,1)
    ratio_unfairness: tuple (min, max) where min is the minimum ratio of samples that we want to remove from the group_to_reduce
        and max is the maximum ratio of samples that we want to remove from the group_to_reduce
    """

    # check if the number of samples that we want in each node is
    # greater than the number of samples we have in the dataset
    if number_of_samples_per_node:
        assert (
            number_of_samples_per_node < len(y) // num_nodes
        ), "Too many samples per node"
    # check if the ratio_fair_nodes is between 0 and 1
    assert ratio_unfair_nodes <= 1, "ratio_unfair_nodes must be less or equal than 1"
    assert ratio_unfair_nodes >= 0, "ratio_unfair_nodes must be greater or equal than 0"
    assert group_to_reduce, "group_to_reduce must be specified"
    assert group_to_increment, "group_to_increment must be specified"
    # check if the approach type is egalitarian or representative
    assert approach in [
        "egalitarian",
        "representative",
    ], "Approach must be egalitarian or representative"

    number_unfair_nodes = int(num_nodes * ratio_unfair_nodes)
    number_fair_nodes = num_nodes - number_unfair_nodes
    if approach == "egalitarian":
        # first split the data among the nodes in an egalitarian way
        # each node has the same amount of data and the same ratio of each group
        nodes, remaining_data = egalitarian_approach(
            X, y, z, num_nodes, number_of_samples_per_node
        )
    else:
        nodes, remaining_data = representative_diversity_approach(
            X, y, z, num_nodes, number_of_samples_per_node
        )

    if opposite_direction:
        assert opposite_group_to_reduce, "opposite_group_to_reduce must be specified"
        assert (
            opposite_group_to_increment
        ), "opposite_group_to_increment must be specified"
        group_size = number_unfair_nodes // 2
        unfair_nodes_direction_1 = create_unfair_nodes(
            nodes_to_unfair=nodes[number_fair_nodes : number_fair_nodes + group_size],
            remaining_data=remaining_data,
            group_to_reduce=group_to_reduce,
            group_to_increment=group_to_increment,
            ratio_unfairness=ratio_unfairness,
        )
        unfair_nodes_direction_2 = create_unfair_nodes(
            nodes_to_unfair=nodes[number_fair_nodes + group_size :],
            remaining_data=remaining_data,
            group_to_reduce=opposite_group_to_reduce,
            group_to_increment=opposite_group_to_increment,
            ratio_unfairness=opposite_ratio_unfairness,
        )
        return (
            nodes[0:number_fair_nodes]
            + unfair_nodes_direction_1
            + unfair_nodes_direction_2
        ), [0] * number_fair_nodes + [1] * len(unfair_nodes_direction_1)
    else:
        unfair_nodes = create_unfair_nodes(
            nodes_to_unfair=nodes[number_fair_nodes:],
            remaining_data=remaining_data,
            group_to_reduce=group_to_reduce,
            group_to_increment=group_to_increment,
            ratio_unfairness=ratio_unfairness,
        )
        return (
            nodes[0:number_fair_nodes] + unfair_nodes,
            [0] * number_fair_nodes + [1] * number_fair_nodes,
        )


# def generate_clients_biased_data(
#     x, y, z, M, do_iid_split, clients_balance_factor, priv_balance_factor
# ):
#     """Args:
#         x : numpy array of non-sensitive features
#         y : numpy array of targets
#         z : numpy array of sensitive feature(s), 1-hot encoded as 2 features: (priv, unpriv) [currently only single sensitive feature supported]
#         M : int > 1, number of clients (should be even)
#         do_iid_split : bool, whether to do iid split or split according to clients_balance_factor and priv_balance_factor
#         clients_balance_factor : float in [0,1], fraction of privileged clients
#         priv_balance_factor : float [.5,1.] fraction of priv samples the privileged clients should have; rest of the samples are divided equally between unprivileged clients
#     Returns:
#         list of dicts with keys x, y corresponding to features and target
#     """
#     # how this should work?
#     # client_balance_factor in [0,1] controls fraction of clients who will have only priv classes, rest will have only unpriv
#     # all priv clients have same number of samples, which depends on total number of priv samples and number of priv clients
#     # target variable split is not controlled separately but depends on priv/unpriv feature
#     # will actually want at least some samples from both priv and unpriv groups on each client, otherwise poisoning is hard
#     # try to populate privileged clients with given frac of priv samples, divide rest equally
#     assert M > 1, "Need more than 1 client!"
#     assert (
#         0.0 <= clients_balance_factor <= 1.0 and 0.5 <= priv_balance_factor <= 1.0
#     ), "Invalid privileged fractions!"
#     N = x.shape[0]
#     # shuffle data to avoid unintential biasing in the target when doing client split
#     print(f"shapes before shuffle: {x.shape}, {y.shape}, {z.shape}")
#     shuffle_inds = np.random.permutation(N)
#     x = x[shuffle_inds, :]
#     y = y[shuffle_inds]
#     z = z[shuffle_inds, :]
#     # check that data is 1-hot encoded
#     assert np.all(
#         len(np.unique(z)) == 2
#     ), f"Sensitive features not properly 1-hot encoded! Got uniques: {np.unique(z)}"
#     assert (
#         z.shape[1] == 2
#     ), "Currently only single 1-hot encoded sensitive feature supported!"
#     # x_priv = x[z==1]
#     # include sensitive features in x
#     x = np.hstack((x, z))
#     # do iid split
#     if do_iid_split:
#         print("Doing iid split!")
#         # shuffle data
#         shuffle_inds = np.random.permutation(N)
#         x = x[shuffle_inds, :]
#         y = y[shuffle_inds]
#         z = z[shuffle_inds, :]
#         # get only the first column of z
#         z = z[:, 0]
#         print(z)
#         # split data into M parts
#         client_data = []
#         for i in range(M):
#             client_x = x[i::M]
#             client_y = y[i::M]
#             client_z = z[i::M]
#             client_data.append({"x": client_x, "y": client_y, "z": client_z})
#         N_is = [data["x"].shape[0] for data in client_data]
#         props_positive = [np.mean(data["y"] > 0) for data in client_data]
#         return client_data, N_is, props_positive
#     # non-iid split
#     M_priv = int(M * clients_balance_factor)
#     M_unpriv = M - M_priv
#     assert (
#         M_priv > 0 and M_unpriv > 0
#     ), f"Got num priv clients={M_priv}, unpriv={M_unpriv}, try changing client balance factor!"
#     print(f"Number of priv clients: {M_priv}, unpriv clients: {M_unpriv}")
#     # group priv/unpriv samples
#     x_priv = x[z[:, 0] == 1]
#     x_unpriv = x[z[:, 1] == 1]
#     y_priv = y[z[:, 0] == 1]
#     y_unpriv = y[z[:, 1] == 1]
#     N_priv = x_priv.shape[0]
#     N_unpriv = x_unpriv.shape[0]
#     print(
#         f"Total number of priv samples: {N_priv} ({np.round(np.sum(y_priv==1)/N_priv,4)} positive label), unpriv samples: {N_unpriv} ({np.round(np.sum(y_unpriv==1)/N_unpriv,4)} positive label)"
#     )
#     priv_priv_client_size = int(N_priv * priv_balance_factor / M_priv)
#     priv_unpriv_client_size = int(N_unpriv * (1 - priv_balance_factor) / M_priv)
#     unpriv_priv_client_size = int((N_priv - M_priv * priv_priv_client_size) / M_unpriv)
#     unpriv_unpriv_client_size = int(
#         (N_unpriv - M_priv * priv_unpriv_client_size) / M_unpriv
#     )
#     assert (
#         priv_priv_client_size > 0
#         and priv_unpriv_client_size > 0
#         and unpriv_priv_client_size > 0
#         and unpriv_unpriv_client_size > 0
#     ), f"Invalid client partitioning: got 0 priv/unpriv divide over clients ({priv_priv_client_size,priv_unpriv_client_size, unpriv_priv_client_size, unpriv_unpriv_client_size})!"
#     print(
#         f"Priv clients size: {priv_priv_client_size}+{priv_unpriv_client_size}, unpriv clients size: {unpriv_priv_client_size}+{unpriv_unpriv_client_size}"
#     )
#     client_data = []
#     # Populate privileged clients.
#     for i in range(M_priv):
#         client_x = np.vstack(
#             (x_priv[:priv_priv_client_size], x_unpriv[:priv_unpriv_client_size])
#         )
#         x_priv = x_priv[priv_priv_client_size:]
#         x_unpriv = x_unpriv[priv_unpriv_client_size:]
#         client_y = np.concatenate(
#             (y_priv[:priv_priv_client_size], y_priv[:priv_unpriv_client_size])
#         )
#         y_priv = y_priv[priv_priv_client_size:]
#         y_unpriv = y_unpriv[priv_unpriv_client_size:]
#         shuffle_inds = np.random.permutation(client_x.shape[0])
#         client_x = client_x[shuffle_inds, :]
#         client_y = client_y[shuffle_inds]
#         client_data.append({"x": client_x, "y": client_y})
#     # Populate unprivileged clients.
#     for i in range(M_unpriv):
#         client_x = np.vstack(
#             (x_priv[:unpriv_priv_client_size], x_unpriv[:unpriv_unpriv_client_size])
#         )
#         x_priv = x_priv[unpriv_priv_client_size:]
#         x_unpriv = x_unpriv[unpriv_unpriv_client_size:]
#         client_y = np.concatenate(
#             (y_priv[:unpriv_priv_client_size], y_unpriv[:unpriv_unpriv_client_size])
#         )
#         y_priv = y_priv[unpriv_priv_client_size:]
#         y_unpriv = y_unpriv[unpriv_unpriv_client_size:]
#         shuffle_inds = np.random.permutation(client_x.shape[0])
#         client_x = client_x[shuffle_inds, :]
#         client_y = client_y[shuffle_inds]
#         client_data.append({"x": client_x, "y": client_y})

#     for client_dataset in client_data:
#         # get the last column
#         current_z = client_dataset["x"][:, -1]
#         client_dataset["z"] = current_z
#     # remove the last two columns from client_dataset["x"]
#     for client_dataset in client_data:
#         client_dataset["x"] = client_dataset["x"][:, :-2]

#     N_is = [data["x"].shape[0] for data in client_data]
#     props_positive = [np.mean(data["y"] > 0) for data in client_data]
#     return client_data, N_is, props_positive


def load_dutch(dataset_path):
    data = arff.loadarff(dataset_path + "dutch_census.arff")
    dutch_df = pd.DataFrame(data[0]).astype("int32")

    dutch_df["sex_binary"] = np.where(dutch_df["sex"] == 1, 1, 0)
    dutch_df["occupation_binary"] = np.where(dutch_df["occupation"] >= 300, 1, 0)

    del dutch_df["sex"]
    del dutch_df["occupation"]

    dutch_df_feature_columns = [
        "age",
        "household_position",
        "household_size",
        "prev_residence_place",
        "citizenship",
        "country_birth",
        "edu_level",
        "economic_status",
        "cur_eco_activity",
        "Marital_status",
        "sex_binary",
    ]

    metadata_dutch = {
        "name": "Dutch census",
        "code": ["DU1"],
        "protected_atts": ["sex_binary"],
        "protected_att_values": [0],
        "protected_att_descriptions": ["Gender = Female"],
        "target_variable": "occupation_binary",
    }

    return dutch_df, dutch_df_feature_columns, metadata_dutch


## Use this function to retrieve X, X, y arrays for training ML models
def dataset_to_numpy(
    _df,
    _feature_cols: list,
    _metadata: dict,
    num_sensitive_features: int = 1,
    sensitive_features_last: bool = True,
):
    """Args:
    _df: pandas dataframe
    _feature_cols: list of feature column names
    _metadata: dictionary with metadata
    num_sensitive_features: number of sensitive features to use
    sensitive_features_last: if True, then sensitive features are encoded as last columns
    """

    # transform features to 1-hot
    _X = _df[_feature_cols]
    # take sensitive features separately
    print(
        f'Using {_metadata["protected_atts"][:num_sensitive_features]} as sensitive feature(s).'
    )
    if num_sensitive_features > len(_metadata["protected_atts"]):
        num_sensitive_features = len(_metadata["protected_atts"])
    _Z = _X[_metadata["protected_atts"][:num_sensitive_features]]
    _X = _X.drop(columns=_metadata["protected_atts"][:num_sensitive_features])

    # 1-hot encode and scale features
    if "dummy_cols" in _metadata.keys():
        dummy_cols = _metadata["dummy_cols"]
    else:
        dummy_cols = None
    _X2 = pd.get_dummies(_X, columns=dummy_cols, drop_first=False)
    esc = MinMaxScaler()
    _X = esc.fit_transform(_X2)

    # current implementation assumes each sensitive feature is binary
    for i, tmp in enumerate(_metadata["protected_atts"][:num_sensitive_features]):
        assert len(_Z[tmp].unique()) == 2, "Sensitive feature is not binary!"

    # 1-hot sensitive features, (optionally) swap ordering so privileged class feature == 1 is always last, preceded by the corresponding unprivileged feature
    _Z2 = pd.get_dummies(_Z, columns=_Z.columns, drop_first=False)
    # print(_Z2.head(), _Z2.shape)
    if sensitive_features_last:
        for i, tmp in enumerate(_Z.columns):
            assert (
                _metadata["protected_att_values"][i] in _Z[tmp].unique()
            ), "Protected attribute value not found in data!"
            if not np.allclose(float(_metadata["protected_att_values"][i]), 0):
                # swap columns
                _Z2.iloc[:, [2 * i, 2 * i + 1]] = _Z2.iloc[:, [2 * i + 1, 2 * i]]
    # change booleans to floats
    # _Z2 = _Z2.astype(float)
    _Z = _Z2.to_numpy()

    _y = _df[_metadata["target_variable"]].values
    return _X, _Z, _y


def get_tabular_numpy_dataset(dataset_name, num_sensitive_features, dataset_path=None):
    if dataset_name == "compas":
        tmp = load_compas()
    elif dataset_name == "adult":
        tmp = load_adult(dataset_path=dataset_path)
    elif dataset_name == "german_credit":
        tmp = load_german()
    elif dataset_name == "kdd":
        tmp = load_kdd()
    elif dataset_name == "dutch":
        tmp = load_dutch(dataset_path=dataset_path)
    elif dataset_name == "bank_marketing":
        tmp = load_bank_marketing()
    elif dataset_name == "credit_card":
        tmp = load_credit_card()
    elif dataset_name == "communities_and_crime":
        tmp = load_communities_and_crime()
    elif dataset_name == "diabetes":
        tmp = load_diabetes()
    elif dataset_name == "ricci":
        tmp = load_ricci()
    elif dataset_name == "oulad":
        tmp = load_oulad()
    elif dataset_name == "maths_student":
        tmp = load_maths_student()
    elif dataset_name == "portuguese_student":
        tmp = load_portuguese_student()
    elif dataset_name == "lsac":
        tmp = load_lsac()
    else:
        raise ValueError("Unknown dataset name!")
    _X, _Z, _y = dataset_to_numpy(*tmp, num_sensitive_features=num_sensitive_features)
    return _X, _Z, _y


import matplotlib.pyplot as plt


# plot the bar plot of the disparities
def plot_bar_plot(title: str, disparities: list, nodes: list):
    plt.figure(figsize=(20, 8))
    plt.bar(range(len(disparities)), disparities)
    plt.xticks(range(len(nodes)), nodes)
    plt.title(title)
    # add a vertical line on xtick=75
    plt.axvline(x=75, color="r", linestyle="--")
    plt.xticks(rotation=90)
    # plt.show()
    # font size x axis
    plt.rcParams.update({"font.size": 10})
    plt.savefig(f"./{title}.png")
    plt.tight_layout()


def prepare_tabular_data(
    dataset_path: str,
    dataset_name: str,
    groups_balance_factor: float,
    priv_balance_factor: float,
    approach: str,
    num_nodes: int,
    ratio_unfair_nodes: float,
    opposite_direction: bool,
    ratio_unfairness: tuple,
    group_to_reduce: tuple = None,
    group_to_increment: tuple = None,
    number_of_samples_per_node: int = None,
    opposite_group_to_reduce: tuple = None,
    opposite_group_to_increment: tuple = None,
    opposite_ratio_unfairness: tuple = None,
    do_iid_split: bool = False,
):
    # client_data, N_is, props_positive = get_tabular_data(
    client_data, disparities, metadata = get_tabular_data(
        num_clients=150,
        do_iid_split=do_iid_split,
        groups_balance_factor=groups_balance_factor,  # fraction of privileged clients ->
        priv_balance_factor=priv_balance_factor,  # fraction of priv samples the privileged clients should have
        dataset_name="dutch",
        num_sensitive_features=1,
        dataset_path=dataset_path,
        approach=approach,
        num_nodes=num_nodes,
        ratio_unfair_nodes=ratio_unfair_nodes,
        opposite_direction=opposite_direction,
        ratio_unfairness=ratio_unfairness,
        group_to_reduce=group_to_reduce,
        group_to_increment=group_to_increment,
        number_of_samples_per_node=number_of_samples_per_node,
        opposite_group_to_reduce=opposite_group_to_reduce,
        opposite_group_to_increment=opposite_group_to_increment,
        opposite_ratio_unfairness=opposite_ratio_unfairness,
    )

    # transform client data so that they are compatiblw with the
    # other functions
    tmp_data = []
    for client in client_data:
        tmp_x = []
        tmp_y = []
        tmp_z = []
        for sample in client:
            tmp_x.append(sample["x"])
            tmp_y.append(sample["y"])
            tmp_z.append(sample["z"])
        tmp_data.append(
            {"x": np.array(tmp_x), "y": np.array(tmp_y), "z": np.array(tmp_z)}
        )
    client_data = tmp_data

    # remove the old files in the data folder
    os.system(f"rm -rf {dataset_path}/federated/*")
    for client_name, (client, client_disparity, client_metadata) in enumerate(
        zip(client_data, disparities, metadata)
    ):
        # Append 1 to each samples

        custom_dataset = TabularDataset(
            x=np.hstack((client["x"], np.ones((client["x"].shape[0], 1)))).astype(
                np.float32
            ),
            z=client["z"],  # .astype(np.float32),
            y=client["y"],  # .astype(np.float32),
        )
        # Create the folder for the user client_name
        os.system(f"mkdir {dataset_path}/federated/{client_name}")
        # store the dataset in the client folder with the name "train.pt"
        torch.save(
            custom_dataset,
            f"{dataset_path}/federated/{client_name}/train.pt",
        )
        # store statistics about the dataset in the same folder
        statistics = Utils.get_dataset_statistics(
            custom_dataset, client_disparity, client_metadata
        )
        with open(
            f"{dataset_path}/federated/{client_name}/metadata.json", "w"
        ) as outfile:
            print(statistics)
            json_object = json.dumps(statistics, indent=4)
            outfile.write(json_object)

    fed_dir = f"{dataset_path}/federated"
    return fed_dir, client_data


if __name__ == "__main__":
    client_data, N_is, props_positive = get_tabular_data(
        num_clients=150,
        do_iid_split=False,
        groups_balance_factor=0.6,  # fraction of privileged clients
        priv_balance_factor=0.7,  # fraction of priv samples the privileged clients should have
        dataset_name="dutch",
        num_sensitive_features=2,
        dataset_path="../../data/Tabular/dutch/",
    )

    # custom_dataset = CustomTabularDataset(
    #     x=client_data[0]["x"], z=client_data[0]["z"], y=client_data[0]["y"]
    # )
    # data_loader = torch.utils.data.DataLoader(
    #     custom_dataset, batch_size=64, shuffle=True
    # )

    # for batch in data_loader:
    #     print(batch)

    # test preprocessing: compare data sets against fairness data survey (https://arxiv.org/abs/2110.00530) (& UCI descriptions when available)
    # tmp = load_compas() # checked, should be ok: preprocessing based on NIPS2017 paper (https://arxiv.org/abs/1704.03354), although some details fixed, not exactly same features as in the paper, closer to AIF360 preprocessing
    #   all 0s prediction acc: 0.829
    #   logistic regresssion acc: 0.843
    # tmp = load_adult() # not checked
    # tmp = load_german() # samples and features match what's in fairness dataset survey,
    #   all 0s/1s prediction acc: 0.7
    #   logistic regresssion acc: 0.771
    # tmp = load_kdd() # not fixed, has problems e.g. with missing values
    # tmp = (
    #     load_dutch()
    # )  # samples and features seem to match the fairness data survey and the original paper
    #   all 0s/1s prediction acc: 0.524
    #   logistic regresssion acc: 0.810
    # tmp = load_bank_marketing() # samples and features seem to match the fairness data survey
    # tmp = load_credit_card() # samples and features seem to match the fairness data survey
    # tmp = load_communities_and_crime() # maybe skip for now, pretty small data set
    # tmp = load_diabetes() # note: originally has client structure, but is not directly available in the data; fairness survey seems to mess up at least one feature (readmitted: missing vs none) compared to UCI description; even when discounting this, the number of remaining samples is some thousands different than reported in the fairness survey
    # leave the rest for later

    # print(tmp[1], tmp[2])
    # tmp = dataset_to_numpy(*tmp, num_sensitive_features=1)

    # print(tmp[0].shape, tmp[1].shape, tmp[2].shape)

    # client_data, N_is, props_positive = generate_clients_biased_data(e
    #     x=tmp[0],
    #     y=tmp[1],
    #     z=tmp[2],
    #     M=150,
    #     do_iid_split=False,
    #     clients_balance_factor=0.7,
    #     priv_balance_factor=0.7,
    # )

    # print(f"sensitive attr mean: {tmp[1].mean(0)}")
    # # print(tmp[0].mean(0), '\n',tmp[0].std(0), '\n', np.amax(tmp[0],0), np.amin(tmp[0],0))

    # # check constant prediction and standard logistic regression baselines
    # print(
    #     f"all 0s/1s prediction acc: {np.amax(((tmp[2] == 0).sum()/len(tmp[2]), (tmp[2] == 1).sum()/len(tmp[2]))) }"
    # )
    # logreg = LogisticRegression().fit(tmp[0], tmp[2])
    # print(f"logistic regresssion acc: {logreg.score(tmp[0], tmp[2])}")
    # print(
    #     f"number of 0s predicted by logistic regression: {(logreg.predict(tmp[0]) == 0).sum()}/{len(tmp[2])}"
    # )
