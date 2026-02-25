import io
import time

import pandas as pd
import requests

urls_to_test = [
    # ProPublica's raw scores dataset 
    "https://raw.githubusercontent.com/propublica/compas-analysis/yes/compas-scores.csv",
    # ProPublica's parsed survival dataset
    "https://raw.githubusercontent.com/propublica/compas-analysis/master/cox-parsed.csv",
    # ProPublica's raw dataset with 2-year recidivism
    "https://raw.githubusercontent.com/propublica/compas-analysis/master/compas-scores-two-years.csv",
    # A common slightly cleaned version used by other fairness papers (AIF360)
    "https://raw.githubusercontent.com/algofairness/fairness-comparison/master/fairness/data/preprocessed/compas.csv"
]

def test_url(url):
    print(f"\n--- Testing URL: {url} ---")
    try:
        # Some URLs might need headers or error handling
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        
        # Read the raw CSV content
        csv_content = io.StringIO(response.text)
        df_raw = pd.read_csv(csv_content)
        print(f"Original dataset size: {len(df_raw)}")
        
        # Now apply the exact filtering from datasets.py
        df = df_raw.copy()
        
        if "days_b_screening_arrest" not in df.columns:
             print("Dataset is missing 'days_b_screening_arrest'. Skiping.")
             return
             
        df = df.dropna(subset=["days_b_screening_arrest"])
        df = df[(df['days_b_screening_arrest']<=30) & (df['days_b_screening_arrest']>=-30)]
        
        if "is_recid" not in df.columns or "c_charge_degree" not in df.columns or "score_text" not in df.columns or "race" not in df.columns:
            print("Dataset is missing one of the filtering columns. Skipping.")
            return

        df = df[df['is_recid']!=-1]
        df = df[df['c_charge_degree']!='O']
        df = df[df['score_text']!='NA']
        df = df[(df['race']=='African-American') | (df['race']=='Caucasian')]
        
        # Reset index
        df = df.reset_index()
        
        final_size = len(df)
        print(f"FINAL PREPROCESSED SIZE: {final_size}")
        
        if final_size == 5278:
            print(">>> MATCH FOUND!! <<<")
            
    except Exception as e:
        print(f"Failed to process {url}: {e}")

for u in urls_to_test:
    test_url(u)
    time.sleep(1)
