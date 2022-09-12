import pickle
import numpy as np
import pandas as pd
import os

from tqdm.notebook import tqdm

from IPython.display import display

RANDOM_STATE = 34
N_CPU = os.cpu_count()

dir_data = 'data/'

try:   
    from lightfm.evaluation import precision_at_k, recall_at_k
except:
    pass


def prepro_users(df):
    df.chit_type = df.chit_type.replace(['нет данных','отсутствует'],'no')
    df.gender = df.gender.replace(['не указан','отсутствует','female'],'f')
    df.gender = df.gender.replace(['male'],'m')
    df.age = df.age.replace(['0','отсутствует'],'30').astype(np.int8)

    def age_cats(x):
        if x > 60:
            return '60'
        elif x > 40:
            return '40'
        elif x > 30:
            return '30'
        elif x > 22:
            return '22'
        elif x > 17:
            return '17'
        else:
            return '14'

    df['ff_age'] = df.age.apply(age_cats)

    df['ff_age'] = 'f1:' + df['ff_age']
    df['ff_gender'] = 'f2:' + df.gender
    df['ff_chit_type'] = 'f3:' + df.chit_type
    
    return df

import pymorphy2

morph = pymorphy2.MorphAnalyzer()

def lemmatize(token):
    return morph.normal_forms(token)[0]

# stopws = ['изд', 'во', 'гос', 'т', 'ун', 'и', 'тип', 'б', 'та', 'ка']
stopws =[]

with open('russian') as f:
    stopws += f.read().split()

# удаление стоп-слов
def clean_by_word(text, filter_small=False, do_lemma=False, max_length = 10000):
    tokens = []
    for token in text.split():
        if filter_small and (len(token) < 3):
            continue

        if do_lemma and (len(token) > 4) and ("а" <= token[0] <= "я") and ("а" <= token[-1] <= "я"):
            token = lemmatize(token)

        if token not in stopws:
            tokens.append(token)

        if len(tokens) >= max_length:
            break

    return " ".join(tokens)

def clean_text(text_col):
    # понижение регистра, знаков препинания, лишних пробелов
    text_col = text_col.str.lower()
    text_col = text_col.str.replace("[^A-Za-zА-Яа-я]+", " ", regex=True).str.strip()
    text_col = text_col.str.replace("\s+", " ", regex=True)

    return text_col

def prerpo_items(df):
    no_title = 'notitle'
    df.title = df.title.replace(['отсутствует'],no_title)
    df.author = df.author.replace(['none','отсутствует'],'ноавт')
    df.izd = df.izd.replace(['none','отсутствует'],'ноизд')
    df.year_izd = df.year_izd.replace(['none','отсутствует',np.nan],'2025')

    df.author = clean_text(df.author)
    df.izd = clean_text(df.izd)
    p(df.izd.nunique())
    df.izd = df.izd.parallel_apply(clean_by_word,args=(False,False))

    df.year_izd = df.year_izd.str.findall('[12]\d\d\d').parallel_apply(lambda x:int(x[0]) if x else 2025)
    df.loc[204668,'year_izd'] = 2017
    df.loc[287590,'year_izd'] = 2018

    def year_cats(x):
        if x > 2015:
            return '2015'
        elif x > 2005:
            return '2005'
        elif x > 1995:
            return '1995'
        elif x > 1970:
            return '1970'
        elif x > 1945:
            return '1945'
        elif x > 1900:
            return '1900'
        else:
            return '1800'

    df['ff_year_izd'] = df.year_izd.parallel_apply(year_cats)

    df['ff_year_izd'] = 'f1:' + df['ff_year_izd']

    clean_title = clean_text(df['title'])
    df['clean_lemma_title'] = clean_title.parallel_apply(clean_by_word,args=(True,True,25))
    df['clean_lemma_title'] = df['clean_lemma_title'].replace('',no_title)

    df['clean_no_lemma_title'] = clean_title.parallel_apply(clean_by_word,args=(False,False,30))
    df['clean_no_lemma_title'] = df['clean_no_lemma_title'].replace('',no_title)

    return df



def generate_lightfm_recs_mapper(model, item_ids, known_items, 
                                 user_features, item_features, N_preds, 
                                 user_mapping, item_inv_mapping, 
                                 num_threads=N_CPU, progress_bar=None):

    def _recs_mapper(user):

        progress_bar.update(1)

        user_id = user_mapping[user]
        recs = model.predict(user_id, item_ids, user_features=user_features, 
                             item_features=item_features, num_threads=num_threads)
        
        additional_N = len(known_items[user_id]) if user_id in known_items else 0

        N = N_preds
        if isinstance(N_preds, dict):
            N = N_preds[user]

        total_N = N + additional_N
        top_cols = np.argpartition(recs, -np.arange(total_N))[-total_N:][::-1]
        
        final_recs = [item_inv_mapping[item] for item in top_cols]
        if additional_N > 0:
            filter_items = known_items[user_id]
            final_recs = [item for item in final_recs if item not in filter_items]
        return final_recs[:N]
    return _recs_mapper

def recommend(model, df, lfm_map, user_feats, it_feats, top_N=20, remove_known_items=True):
    sub = pd.read_csv(dir_data+'sample_solution.csv',sep=';')

    users = pd.DataFrame({'chb': sub['chb'].unique()})
    all_items = list(lfm_map['i_to_ids'].values())

    known_user_items = dict()
    if remove_known_items:
        t = df[['chb','sys_numb']].copy()
        t['chb'] = t['chb'].map(lfm_map['u_to_ids'])
        known_user_items = t.groupby('chb')['sys_numb'].agg(list).to_dict()

    pbar = tqdm(total=len(lfm_map['u_to_ids']))

    mapper = generate_lightfm_recs_mapper(
        model, 
        item_ids=all_items, 
        known_items=known_user_items,
        N_preds=top_N,
        user_features=user_feats,
        item_features=it_feats,
        user_mapping=lfm_map['u_to_ids'],
        item_inv_mapping=lfm_map['ids_to_i'],
        num_threads=N_CPU,
        progress_bar=pbar
    )

    users['sys_numb'] = users['chb'].map(mapper)
    pbar.close()

    users = users.explode('sys_numb')



    return users 


def calc_metrics(model, test_int_mat, train_int_mat, user_feats, it_feats, K=20):
    prec_K = precision_at_k(model, test_int_mat,train_int_mat, user_features=user_feats, item_features=it_feats, k=K,num_threads=N_CPU).mean()

    recall_K = recall_at_k(model, test_int_mat,train_int_mat, user_features=user_feats, item_features=it_feats, k=K,num_threads=N_CPU).mean()

    f1_score = 2 * (prec_K * recall_K) / (prec_K + recall_K)

    return (prec_K,recall_K,f1_score)

def p(*args):
    for i, a in enumerate(args):
        if isinstance(a, (pd.Series, pd.DataFrame)):
            display(a)
        else:
            print(a, end='')

        if i < len(args) - 1:
            try:
                len(a)
                print("\n ~")
            except:
                print(" | ", end='')
    print()


def load_pickle(file_name):
    with open(file_name, 'rb') as f:
        obj = pickle.load(f)
    return obj


def save_pickle(file_name, data, verbose = False):
    if verbose:
        print('save: ',file_name)
    with open(file_name, 'wb') as f:
        pickle.dump(data, f)


def select_cols(df, names):
    if not isinstance(names, list):
        names = [names]
    new_cols = []
    for col in df.columns:
        for name in names:
            if col.startswith(name):
                new_cols.append(col)
    return new_cols


def flat_cols(df, pre='k', columns=True):

    def f(se):
        return [
            pre + '_' + '_'.join(map(str, col)) if type(col) is tuple else pre + '_' +
            str(col) for col in se.to_numpy()
        ]

    if columns:
        df.columns = f(df.columns)
    else:
        df.index = f(df.index)


def mem(df):
    memory = df.memory_usage().sum() / 1024**2
    print(f'Память: {round(memory)} Мб')


def reduce_mem(df):
    start_mem = df.memory_usage().sum() / 1024**2
    for col in df.columns:
        col_type = df[col].dtypes
        if str(col_type)[:5] == "float":
            c_min = df[col].min()
            c_max = df[col].max()
            if c_min > np.finfo("f2").min and c_max < np.finfo("f2").max:
                # np.float16 не принимает бывает
                df[col] = df[col].astype(np.float32)
            elif c_min > np.finfo("f4").min and c_max < np.finfo("f4").max:
                df[col] = df[col].astype(np.float32)
            else:
                df[col] = df[col].astype(np.float64)
        elif str(col_type)[:3] == "int":
            c_min = df[col].min()
            c_max = df[col].max()
            if c_min > np.iinfo("i1").min and c_max < np.iinfo("i1").max:
                df[col] = df[col].astype(np.int8)
            elif c_min > np.iinfo("i2").min and c_max < np.iinfo("i2").max:
                df[col] = df[col].astype(np.int16)
            elif c_min > np.iinfo("i4").min and c_max < np.iinfo("i4").max:
                df[col] = df[col].astype(np.int32)
            elif c_min > np.iinfo("i8").min and c_max < np.iinfo("i8").max:
                df[col] = df[col].astype(np.int64)
        # elif col == "timestamp":
        #     df[col] = pd.to_datetime(df[col])
        # elif str(col_type)[:8] != "datetime":
        #     df[col] = df[col].astype("category")
    end_mem = df.memory_usage().sum() / 1024**2
    print(f'Память ДО: {round(start_mem,1)} Мб')
    print(f'Память ПОСЛЕ: {round(end_mem,1)} Мб')
    print('Уменьшилось на', round(start_mem - end_mem, 2), 'Мб (минус',
          round(100 * (start_mem - end_mem) / start_mem, 1), '%)')
    return
