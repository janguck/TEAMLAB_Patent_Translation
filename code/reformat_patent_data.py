import pandas as pd
from sklearn.model_selection import train_test_split

source_lang = 'en'
target_lang = 'ko'
use_rows = 100

def create_pd_dataset(dataset_type, X, Y, source_lang, target_lang, minimum=False, use_rows=99999999999999):
    print(f'************************* [{dataset_type}] Create Complete *************************')
    df = pd.DataFrame({source_lang:X, target_lang:Y})
    file_name = f'../data/patent_{dataset_type}_{source_lang}_{target_lang}.csv'
    if minimum:
        print(f'************************* [{dataset_type}] Small Create Complete *************************')
        df = df[:use_rows]
        file_name = f'../data/small_patent_{dataset_type}_{source_lang}_{target_lang}.csv'
    print(f'{dataset_type} Rows : {len(df)}')
    df.to_csv(file_name, index=False)
    print(f'************************* [{dataset_type}] Write Complete *************************')

file_dir = '../data/en_ko.tsv'

df = pd.read_csv(file_dir, sep='\t', error_bad_lines=False, names=[source_lang, target_lang])

X, Y = df[source_lang].values, df[target_lang].values

train_X, valid_X, train_Y, valid_Y = train_test_split(X, Y, test_size=0.2, random_state=42)
valid_X, test_X, valid_Y, test_Y = train_test_split(valid_X, valid_Y, test_size=0.5, random_state=42)

create_pd_dataset('train', train_X, train_Y, source_lang, target_lang)
create_pd_dataset('validation', valid_X, valid_Y, source_lang, target_lang)
create_pd_dataset('test', test_X, test_Y, source_lang, target_lang)

create_pd_dataset('train', train_X, train_Y, source_lang, target_lang, minimum=True, use_rows=use_rows)
create_pd_dataset('validation', valid_X, valid_Y, source_lang, target_lang, minimum=True, use_rows=use_rows)
create_pd_dataset('test', test_X, test_Y, source_lang, target_lang, minimum=True, use_rows=use_rows)