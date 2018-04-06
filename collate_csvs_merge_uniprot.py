import numpy as np
import pandas as pd
import glob
from tqdm import tqdm

# get all csv files
csv_files = glob.glob('/root/aics/modeling/gregj/projects/seq2loc/data/hpa/*/info.csv')

# read them all as dfs and concat them
df_list = []
for file in tqdm(csv_files):
    df_list += [pd.read_csv(file)]
df = pd.concat(df_list)
df.to_csv('hpa_info_alldat.csv')

# read the uniprot tsv
df_uniprot = pd.read_csv('/root/aics/modeling/gregj/projects/seq2loc/data/uniprot.tsv', sep = '\t')

# rename the protein id cols to match
df = df.rename(index=str, columns={'proteinName': 'protID'})
df_uniprot = df_uniprot.rename(index=str, columns={'Entry': 'protID'})

# merge the dfs and drop some columns we don't care about
df_merged = df.merge(df_uniprot, on='protID', how='left')
df_merged = df_merged.drop(['Unnamed: 0', 'Status', 'Ensembl transcript', 'Organism', 'ENSP', 'Gene names'], axis=1)

# drop rows where we have NaNs in important columns
df_cleaned = df_merged.dropna(subset=['protID', 'antigenSequence', 'Sequence'])

# save to csv
df_cleaned.to_csv('/root/aics/modeling/gregj/projects/seq2loc/data/hpa_data_noNaNs.csv')

