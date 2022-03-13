

import glob

from pandas import read_csv
def uppercase_eng(text):
    return text.replace(' i ', ' I ')

for p in glob.glob('/Users/10972/Documents/NLP_PJ/training/dataset/*.csv'):
    print(p)
    try:
        df = read_csv(p)
        df['sentence'] = df['sentence'].apply(uppercase_eng)
        df.to_csv(p)
    except Exception:
        print('pass', p)