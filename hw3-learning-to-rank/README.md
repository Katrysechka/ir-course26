# ДЗ 3. Обучение ранжированию

Kaggle-ник: Ekaterina Samsanovich, katrussechka

Итоговый скор: 0.62368

CatBoostRanker, без ансамбля и новых признаков. Параметры: `loss_function='YetiRank:mode=NDCG;top=10;dcg_type=Base;dcg_denominator=LogPosition'`, `iterations=600`, `depth=5`, `learning_rate=0.05`, `random_seed=42`, `task_type='CPU'`, `thread_count=4`, `allow_writing_files=False`

Подготовка окружения:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

```bash
./solution.py --train --model_file=model.bin PATH-TO-DATA
./solution.py --model_file=model.bin --submission_file=submission.csv PATH-TO-DATA
```
