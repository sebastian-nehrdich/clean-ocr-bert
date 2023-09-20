import pandas as pd
import os 
import random
import re
from nltk.tokenize import sent_tokenize
path = "."
files = os.listdir(path)

files_txt = [i for i in files if i.endswith('.txt')]

dfs = []
for file in files_txt:
    # read file into string
    with open(file, 'r') as f:
        text = f.read()
    # remove newlines from string
    text = text.replace('\n', ' ')
    # remove all " and '
    text = text.replace('"', '')
    text = text.replace("'", '')
    # remove all numbers    
    sentences = sent_tokenize(text, language='english')
        

    label = 1
    if "note" in file:
        label = 0
    df = pd.DataFrame()
    df['text'] = sentences
    df['label'] = label
    dfs.append(df)

df = pd.concat(dfs)
df = df[df['text'] != '']
df = df.dropna()
# make sure that labels are balanced
positive_samples = df[df['label'] == 1]
negative_samples = df[df['label'] == 0]

min_samples = min(len(positive_samples), len(negative_samples))

balanced_positive = positive_samples.sample(min_samples)
balanced_negative = negative_samples.sample(min_samples)

df = pd.concat([balanced_positive, balanced_negative])

df = df.sample(frac=1).reset_index(drop=True)

df_val = df.iloc[:1000]
df_test = df.iloc[1000:2000]
df_train = df.iloc[2000:]

df_val.to_csv('val.tsv', sep='\t', index=False, header=False)
df_test.to_csv('test.tsv', sep='\t', index=False, header=False)
df_train.to_csv('train.tsv', sep='\t', index=False, header=False)

