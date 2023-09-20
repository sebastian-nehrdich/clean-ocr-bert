import pandas as pd
import os 
import random

path = "."
files = os.listdir(path)

files_txt = [i for i in files if i.endswith('.txt')]



def read_file_to_df(path):    
    df = pd.DataFrame(columns=['sentence_a', 'sentence_b', 'label'])
    last_sentence = 'babababa'
    sentences_a = []
    sentences_b = []
    sentence = ""
    with open(path, 'r') as f:
        for line in f:
            if len(sentence) > 5:
                sentences_a.append(last_sentence)
                sentences_b.append(sentence)
                last_sentence = sentence
                sentence = line.strip()
            else:
                sentence += line.strip()
    df['sentence_a'] = sentences_a
    df['sentence_b'] = sentences_b
    df['label'] = 1
    return df


dfs = []
for file in files_txt:
    df = read_file_to_df(file)
    dfs.append(df)
    
df = pd.concat(dfs, ignore_index=True)

# create a copy of df and shuffle the order of sentence_b, set label to 0
df2 = df.copy()
sentences_b = df2['sentence_b'].tolist()
random.shuffle(sentences_b)
df2['sentence_b'] = sentences_b
df2['label'] = 0

# concat df and df2
df = pd.concat([df, df2], ignore_index=True)

# merge columns sentence_a and sentence_b together with # as separator
df['sentence'] = df['sentence_a'] + ' # ' + df['sentence_b']
del df['sentence_a']
del df['sentence_b']

df = df.sample(frac=1).reset_index(drop=True)

# order of columns: sentence, label
df = df[['sentence', 'label']]
# first 1000 rows for testing, next 1000 vor validation, rest for training
df_test = df.iloc[:1000]
df_val = df.iloc[1000:2000]
df_train = df.iloc[2000:]

df_test.to_csv("./test.tsv", sep='\t', index=False, header=False)
df_val.to_csv("./val.tsv", sep='\t', index=False, header=False)
df_train.to_csv("./train.tsv", sep='\t', index=False, header=False)
