#!/usr/bin/env python
import argparse
import os
import random
from timeit import default_timer as timer

import numpy as np
import pandas as pd
from catboost import CatBoostRanker, Pool


SEED = 42
MODEL_PARAMS = {'allow_writing_files': False,
 'depth': 5,
 'iterations': 600,
 'learning_rate': 0.05,
 'loss_function': 'YetiRank:mode=NDCG;top=10;dcg_type=Base;dcg_denominator=LogPosition',
 'random_seed': 42,
 'task_type': 'CPU',
 'thread_count': 4}


def split_queries(data):
    qids = sorted(data['QueryId'].unique().tolist())
    random.Random(SEED).shuffle(qids)
    n_valid = max(1, min(len(qids) - 1, round(0.2 * len(qids))))
    valid_ids = set(qids[:n_valid])
    train_part = data[~data['QueryId'].isin(valid_ids)]
    valid_part = data[data['QueryId'].isin(valid_ids)]
    return tuple(part.sort_values('QueryId', kind='stable').reset_index(drop=True)
                 for part in (train_part, valid_part))

def make_pool(data, features):
    return Pool(data[features], label=data['Label'], group_id=data['QueryId'])


def mean_query_ndcg(data, scores, k=10):
    scores = np.asarray(scores, dtype=float)
    frame = data[['QueryId', 'Label']].copy()
    frame['Score'] = scores
    values = []
    for _, group in frame.groupby('QueryId', sort=False):
        labels = group['Label'].to_numpy(dtype=float)
        order = np.argsort(-group['Score'].to_numpy(), kind='stable')
        limit = min(k, len(labels))
        discount = np.log2(np.arange(limit) + 2)
        ideal = np.sort(labels)[::-1][:limit] / discount
        actual = labels[order][:limit] / discount
        values.append(float(actual.sum() / ideal.sum()) if ideal.sum() > 0 else 0.0)
    return float(np.mean(values))


def train(data_dir, model_file):
    data = pd.read_csv(os.path.join(data_dir, 'train.csv'))
    features = [column for column in data.columns if column not in ('QueryId', 'Label')]

    train_part, valid_part = split_queries(data)
    model = CatBoostRanker(**MODEL_PARAMS)
    model.fit(make_pool(train_part, features), verbose=50)
    scores = model.predict(valid_part[features])
    print(f'holdout nDCG@10 = {mean_query_ndcg(valid_part, scores):.6f}')

    data = data.sort_values('QueryId', kind='stable').reset_index(drop=True)
    final_model = CatBoostRanker(**MODEL_PARAMS)
    final_model.fit(make_pool(data, features), verbose=50)
    final_model.save_model(model_file)


def predict(data_dir, model_file, submission_file):
    test = pd.read_csv(os.path.join(data_dir, 'test.csv'))
    features = [column for column in test.columns if column not in ('ObjectId', 'QueryId')]

    model = CatBoostRanker()
    model.load_model(model_file)
    scores = model.predict(test[model.feature_names_])
    if not np.isfinite(scores).all():
        raise ValueError('Модель вернула NaN или бесконечные скоры')

    submission = pd.DataFrame({'ObjectId': test['ObjectId'], 'Score': scores})
    submission.to_csv(submission_file, index=False)


def main():
    parser = argparse.ArgumentParser(description='Learning to rank homework solution')
    parser.add_argument('--train', action='store_true', help='обучить модель и сохранить её в model_file')
    parser.add_argument('--model_file', required=True, help='файл модели')
    parser.add_argument('--submission_file', help='куда записать сабмишн для Kaggle')
    parser.add_argument('data_dir', help='папка с данными соревнования')
    args = parser.parse_args()

    start = timer()

    if args.train:
        train(args.data_dir, args.model_file)
    else:
        if args.submission_file is None:
            parser.error('без --train нужен --submission_file')
        predict(args.data_dir, args.model_file, args.submission_file)

    print(f'finished, elapsed = {timer() - start:.3f}')


if __name__ == '__main__':
    main()
