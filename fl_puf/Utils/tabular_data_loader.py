import os

import numpy as np
import pandas as pd
import torch
from scipy.io import arff
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import MinMaxScaler

from DPL.Datasets.dutch import TabularDataset

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


def load_adult():
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

    df_adult = pd.read_csv("data/adult.data", names=adult_columns_names)
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


#########################################################################################


def get_tabular_data(
    num_clients: int,
    do_iid_split: bool,
    groups_balance_factor: float,
    priv_balance_factor: float,
    dataset_name: str,
    num_sensitive_features: int,
    dataset_path=None,
):
    X, z, y = get_tabular_numpy_dataset(
        dataset_name=dataset_name,
        num_sensitive_features=num_sensitive_features,
        dataset_path=dataset_path,
    )
    print(f"Data shapes: x={X.shape}, y={y.shape}, z={z.shape}")
    # Prepare training data held by each client
    client_data, N_is, props_positive = generate_clients_biased_data(
        x=X,
        y=y,
        z=z,
        M=num_clients,
        do_iid_split=do_iid_split,
        clients_balance_factor=groups_balance_factor,
        priv_balance_factor=priv_balance_factor,
    )
    return client_data, N_is, props_positive


def generate_clients_biased_data(
    x, y, z, M, do_iid_split, clients_balance_factor, priv_balance_factor
):
    """Args:
        x : numpy array of non-sensitive features
        y : numpy array of targets
        z : numpy array of sensitive feature(s), 1-hot encoded as 2 features: (priv, unpriv) [currently only single sensitive feature supported]
        M : int > 1, number of clients (should be even)
        do_iid_split : bool, whether to do iid split or split according to clients_balance_factor and priv_balance_factor
        clients_balance_factor : float in [0,1], fraction of privileged clients
        priv_balance_factor : float [.5,1.] fraction of priv samples the privileged clients should have; rest of the samples are divided equally between unprivileged clients
    Returns:
        list of dicts with keys x, y corresponding to features and target
    """
    # how this should work?
    # client_balance_factor in [0,1] controls fraction of clients who will have only priv classes, rest will have only unpriv
    # all priv clients have same number of samples, which depends on total number of priv samples and number of priv clients
    # target variable split is not controlled separately but depends on priv/unpriv feature
    # will actually want at least some samples from both priv and unpriv groups on each client, otherwise poisoning is hard
    # try to populate privileged clients with given frac of priv samples, divide rest equally
    assert M > 1, "Need more than 1 client!"
    assert (
        0.0 <= clients_balance_factor <= 1.0 and 0.5 <= priv_balance_factor <= 1.0
    ), "Invalid privileged fractions!"
    N = x.shape[0]
    # shuffle data to avoid unintential biasing in the target when doing client split
    print(f"shapes before shuffle: {x.shape}, {y.shape}, {z.shape}")
    shuffle_inds = np.random.permutation(N)
    x = x[shuffle_inds, :]
    y = y[shuffle_inds]
    z = z[shuffle_inds, :]
    # check that data is 1-hot encoded
    assert np.all(
        len(np.unique(z)) == 2
    ), f"Sensitive features not properly 1-hot encoded! Got uniques: {np.unique(z)}"
    assert (
        z.shape[1] == 2
    ), "Currently only single 1-hot encoded sensitive feature supported!"
    # x_priv = x[z==1]
    # include sensitive features in x
    x = np.hstack((x, z))
    # do iid split
    if do_iid_split:
        print("Doing iid split!")
        # shuffle data
        shuffle_inds = np.random.permutation(N)
        x = x[shuffle_inds, :]
        y = y[shuffle_inds]
        # split data into M parts
        client_data = []
        for i in range(M):
            client_x = x[i::M]
            client_y = y[i::M]
            client_data.append({"x": client_x, "y": client_y})
        N_is = [data["x"].shape[0] for data in client_data]
        props_positive = [np.mean(data["y"] > 0) for data in client_data]
        return client_data, N_is, props_positive
    # non-iid split
    M_priv = int(M * clients_balance_factor)
    M_unpriv = M - M_priv
    assert (
        M_priv > 0 and M_unpriv > 0
    ), f"Got num priv clients={M_priv}, unpriv={M_unpriv}, try changing client balance factor!"
    print(f"Number of priv clients: {M_priv}, unpriv clients: {M_unpriv}")
    # group priv/unpriv samples
    x_priv = x[z[:, 0] == 1]
    x_unpriv = x[z[:, 1] == 1]
    y_priv = y[z[:, 0] == 1]
    y_unpriv = y[z[:, 1] == 1]
    N_priv = x_priv.shape[0]
    N_unpriv = x_unpriv.shape[0]
    print(
        f"Total number of priv samples: {N_priv} ({np.round(np.sum(y_priv==1)/N_priv,4)} positive label), unpriv samples: {N_unpriv} ({np.round(np.sum(y_unpriv==1)/N_unpriv,4)} positive label)"
    )
    priv_priv_client_size = int(N_priv * priv_balance_factor / M_priv)
    priv_unpriv_client_size = int(N_unpriv * (1 - priv_balance_factor) / M_priv)
    unpriv_priv_client_size = int((N_priv - M_priv * priv_priv_client_size) / M_unpriv)
    unpriv_unpriv_client_size = int(
        (N_unpriv - M_priv * priv_unpriv_client_size) / M_unpriv
    )
    assert (
        priv_priv_client_size > 0
        and priv_unpriv_client_size > 0
        and unpriv_priv_client_size > 0
        and unpriv_unpriv_client_size > 0
    ), f"Invalid client partitioning: got 0 priv/unpriv divide over clients ({priv_priv_client_size,priv_unpriv_client_size, unpriv_priv_client_size, unpriv_unpriv_client_size})!"
    print(
        f"Priv clients size: {priv_priv_client_size}+{priv_unpriv_client_size}, unpriv clients size: {unpriv_priv_client_size}+{unpriv_unpriv_client_size}"
    )
    client_data = []
    # Populate privileged clients.
    for i in range(M_priv):
        client_x = np.vstack(
            (x_priv[:priv_priv_client_size], x_unpriv[:priv_unpriv_client_size])
        )
        x_priv = x_priv[priv_priv_client_size:]
        x_unpriv = x_unpriv[priv_unpriv_client_size:]
        client_y = np.concatenate(
            (y_priv[:priv_priv_client_size], y_priv[:priv_unpriv_client_size])
        )
        y_priv = y_priv[priv_priv_client_size:]
        y_unpriv = y_unpriv[priv_unpriv_client_size:]
        shuffle_inds = np.random.permutation(client_x.shape[0])
        client_x = client_x[shuffle_inds, :]
        client_y = client_y[shuffle_inds]
        client_data.append({"x": client_x, "y": client_y})
    # Populate unprivileged clients.
    for i in range(M_unpriv):
        client_x = np.vstack(
            (x_priv[:unpriv_priv_client_size], x_unpriv[:unpriv_unpriv_client_size])
        )
        x_priv = x_priv[unpriv_priv_client_size:]
        x_unpriv = x_unpriv[unpriv_unpriv_client_size:]
        client_y = np.concatenate(
            (y_priv[:unpriv_priv_client_size], y_unpriv[:unpriv_unpriv_client_size])
        )
        y_priv = y_priv[unpriv_priv_client_size:]
        y_unpriv = y_unpriv[unpriv_unpriv_client_size:]
        shuffle_inds = np.random.permutation(client_x.shape[0])
        client_x = client_x[shuffle_inds, :]
        client_y = client_y[shuffle_inds]
        client_data.append({"x": client_x, "y": client_y})

    for client_dataset in client_data:
        # get the last column
        current_z = client_dataset["x"][:, -1]
        client_dataset["z"] = current_z
    # remove the last two columns from client_dataset["x"]
    for client_dataset in client_data:
        client_dataset["x"] = client_dataset["x"][:, :-2]

    N_is = [data["x"].shape[0] for data in client_data]
    props_positive = [np.mean(data["y"] > 0) for data in client_data]
    return client_data, N_is, props_positive


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
        tmp = load_adult()
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


def prepare_tabular_data(
    dataset_path: str,
    dataset_name: str,
    groups_balance_factor: float,
    priv_balance_factor: float,
):
    client_data, N_is, props_positive = get_tabular_data(
        num_clients=150,
        do_iid_split=False,
        groups_balance_factor=groups_balance_factor,  # fraction of privileged clients ->
        priv_balance_factor=priv_balance_factor,  # fraction of priv samples the privileged clients should have
        dataset_name="dutch",
        num_sensitive_features=1,
        dataset_path=dataset_path,
    )
    # remove the old files in the data folder
    os.system(f"rm -rf {dataset_path}/federated/*")
    for client_name, client in enumerate(client_data):
        # Append 1 to each samples

        custom_dataset = TabularDataset(
            x=np.hstack((client["x"], np.ones((client["x"].shape[0], 1)))).astype(
                np.float32
            ),
            z=client["z"].astype(np.float32),
            y=client["y"].astype(np.float32),
        )
        # Create the folder for the user client_name
        os.system(f"mkdir {dataset_path}/federated/{client_name}")
        # store the dataset in the client folder with the name "train.pt"
        torch.save(
            custom_dataset,
            f"{dataset_path}/federated/{client_name}/train.pt",
        )
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
