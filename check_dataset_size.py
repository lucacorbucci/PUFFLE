import pandas as pd

def test_file(filename):
    try:
        df = pd.read_csv(filename)
        # filtering exactly from datasets.py
        df = df.dropna(subset=["days_b_screening_arrest"])
        df = df[(df['days_b_screening_arrest']<=30) & (df['days_b_screening_arrest']>=-30)]
        df = df[df['is_recid']!=-1]
        df = df[df['c_charge_degree']!='O']
        df = df[df['score_text']!='NA']
        df = df[(df['race']=='African-American') | (df['race']=='Caucasian')]
        df = df.reset_index()
        print(f"Final dataset size for {filename}: {len(df)}")
    except Exception as e:
        print(f"Failed to process {filename}: {e}")

test_file('cox-parsed.csv')
