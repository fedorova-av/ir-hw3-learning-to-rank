# ДЗ 3. Обучение ранжированию

**Ник на Kaggle:** [Fedorova Alexandra](https://www.kaggle.com/kassandra23)

**Скор на публичном лидерборде (nDCG@10):** 0.62607  
**Бейзлайн:** 0.61621

## Лучшие гиперпараметры

Модель - `CatBoostRanker`:

- `loss_function`: `YetiRank`
- `iterations`: `964`
- `depth`: `8`
- `learning_rate`: `0.03`
- `l2_leaf_reg`: `3.0`
- `random_seed`: `42`
- `task_type`: `CPU`

Используется 141 признак: 67 исходных и дополнительные нормировки и ранги внутри запроса.