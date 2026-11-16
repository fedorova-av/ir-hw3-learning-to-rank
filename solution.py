#!/usr/bin/env python
# ДЗ 3. Обучение ранжированию — шаблон решения.
#
# Обучение:  ./solution.py --train --model_file=model.bin PATH-TO-DATA
# Сабмишн:   ./solution.py --model_file=model.bin --submission_file=submission.csv PATH-TO-DATA

import argparse
import json
import os
from timeit import default_timer as timer

import numpy as np
import pandas as pd
from catboost import CatBoostRanker


SEED = 42
ITERATIONS = 964
CONFIG_KEY = "hw3_feature_config"


def make_features(data, features, transform_columns):
    parts = [data[features]]

    if transform_columns:
        grouped = data.groupby("QueryId", sort=False)[transform_columns]

        means = grouped.transform("mean")
        stds = grouped.transform("std").replace(0, 1).fillna(1)
        normalized = (data[transform_columns] - means) / stds

        sizes = data.groupby("QueryId", sort=False)["QueryId"].transform("size")
        ranks = grouped.rank(method="average", ascending=False)
        ranks = (ranks - 1).div((sizes - 1).clip(lower=1), axis=0)

        parts.extend([normalized.add_suffix("__q_z"), ranks.add_suffix("__q_rank")])

    return pd.concat(parts, axis=1).astype(np.float32)

def train(data_dir, model_file):
    data = pd.read_csv(os.path.join(data_dir, "train.csv"))
    features = [column for column in data.columns if column not in ("QueryId", "Label")]

    data = data.sort_values("QueryId", kind="stable").reset_index(drop=True)

    transform_columns = [column for column in features
                         if not column.startswith("q_")
                         and not column.endswith(("_rank", "_to_max", "_minus_mean"))
                         and data[column].nunique() > 2]

    model = CatBoostRanker(
        loss_function="YetiRank",
        iterations=ITERATIONS,
        depth=8,
        learning_rate=0.03,
        l2_leaf_reg=3.0,
        random_seed=SEED,
        eval_metric="NDCG:top=10;type=Exp;denominator=LogPosition",
        task_type="CPU",
        thread_count=min(8, os.cpu_count() or 1),
        allow_writing_files=False,
        verbose=False)

    model.fit(
        make_features(data, features, transform_columns),
        data["Label"],
        group_id=data["QueryId"],
        use_best_model=False)

    config = {"base_features": features, "transform_columns": transform_columns}
    model.get_metadata()[CONFIG_KEY] = json.dumps(config)
    model.save_model(model_file, format="cbm")

def predict(data_dir, model_file, submission_file):
    test = pd.read_csv(os.path.join(data_dir, "test.csv"))

    model = CatBoostRanker()
    model.load_model(model_file, format="cbm")
    config = json.loads(model.get_metadata()[CONFIG_KEY])

    features = make_features(test, config["base_features"], config["transform_columns"])

    if features.columns.tolist() != model.feature_names_:
        raise ValueError("Порядок признаков не совпадает с моделью")

    scores = model.predict(features)

    if not np.isfinite(scores).all():
        raise ValueError("Некорректные предсказания")

    submission = pd.DataFrame({"ObjectId": test["ObjectId"], "Score": scores})
    submission.to_csv(submission_file, index=False)


def main():
    parser = argparse.ArgumentParser(description="Learning to rank homework solution")
    parser.add_argument("--train", action="store_true", help="обучить модель")
    parser.add_argument("--model_file", required=True, help="файл модели")
    parser.add_argument("--submission_file", help="файл сабмишна")
    parser.add_argument("data_dir", help="папка с данными соревнования")
    args = parser.parse_args()

    start = timer()

    if args.train:
        train(args.data_dir, args.model_file)
    else:
        if args.submission_file is None:
            parser.error("без --train нужен --submission_file")
        predict(args.data_dir, args.model_file, args.submission_file)

    print(f"finished, elapsed = {timer() - start:.3f}")


if __name__ == "__main__":
    main()
