import pandas as pd

filename = '/home/lcorbucci/PUFFLE/Fair-FL/datasets/compas-scores-two-years.csv'
df = pd.read_csv(filename, nrows=5)
print("Columns in the new dataset:")
print(list(df.columns))
