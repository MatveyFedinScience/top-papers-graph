# Отчёт об оптимизации Qwen3-VL-8B SFT+GRPO под Yandex DataSphere и бюджет 100 000 ₽

Дата: 2026-05-17.

## Ключевое исправление

Пайплайн переведён с неверной постановки `imagefolder -> label` на фактический HF export:

```text
top-papers/top-papers-graph-experts-data/exports/colab-run-001/sft.jsonl
top-papers/top-papers-graph-experts-data/exports/colab-run-001/grpo.jsonl
```

Теперь `build_hf_graph_experts_dataset.py` скачивает export через `snapshot_download`, разрешает относительные пути к изображениям относительно snapshot root, ограничивает число изображений на пример, делает stratified train/eval split по `task_family` и сохраняет audit summary.

## Бюджет

Выбран `g2.2`: 2 x A100, 160 GB VRAM суммарно. Цена, используемая в guard: `1085.76 ₽/час`.

Настройки:

```text
DATA_TIMEOUT_HOURS=3
SFT_TIMEOUT_HOURS=48
GRPO_TIMEOUT_HOURS=35
HF_UPLOAD_TIMEOUT_HOURS=1
TOTAL=87 часов
```

Worst-case compute:

```text
87 * 1085.76 = 94461.12 ₽
```

Резерв:

```text
BUDGET_RESERVE_RUB=5000 ₽
PROJECTED_TOTAL=99461.12 ₽
```

Остаток до лимита:

```text
538.88 ₽
```

В runtime wrapper добавлен глобальный budget deadline:

```text
((BUDGET_RUB - BUDGET_RESERVE_RUB) / G2_2_RUB_PER_HOUR * 3600) - BUDGET_SHUTDOWN_MARGIN_SECONDS
```

## Изменённые файлы

- `experiments/vlm_finetuning/scripts/build_hf_graph_experts_dataset.py`
- `experiments/vlm_finetuning/scripts/train_vlm_sft.py`
- `experiments/vlm_finetuning/scripts/train_vlm_grpo.py`
- `experiments/vlm_finetuning/scripts/upload_hf_finetuned_artifacts.py`
- `experiments/vlm_finetuning/scripts/estimate_datasphere_costs.py`
- `experiments/vlm_finetuning/datasphere/bin/run_hf_top_papers_sft_grpo_full.sh`
- `experiments/vlm_finetuning/datasphere/job_configs/hf_top_papers_sft_grpo_full_g2_2.yaml`
- `experiments/vlm_finetuning/configs/sft_grpo_full_qwen3vl_8b_lora.yaml`
- `experiments/vlm_finetuning/datasphere/requirements.txt`
- `tests/test_datasphere_job_configs.py`
- `TUTORIAL_QWEN3VL_DATASPHERE_BUDGET_RU.md`

## Проверки

Выполнено локально:

```bash
python -m py_compile \
  experiments/vlm_finetuning/scripts/build_hf_graph_experts_dataset.py \
  experiments/vlm_finetuning/scripts/train_vlm_sft.py \
  experiments/vlm_finetuning/scripts/train_vlm_grpo.py \
  experiments/vlm_finetuning/scripts/upload_hf_finetuned_artifacts.py \
  experiments/vlm_finetuning/scripts/estimate_datasphere_costs.py \
  experiments/vlm_finetuning/datasphere/run_full_pipeline.py

bash -n experiments/vlm_finetuning/datasphere/bin/run_hf_top_papers_sft_grpo_full.sh
for f in experiments/vlm_finetuning/datasphere/bin/*.sh; do bash -n "$f"; done
python -m pytest -q tests/test_datasphere_job_configs.py tests/test_vlm_training_format_normalization.py
```

Результат:

```text
6 passed
```

Фактическое GPU-обучение не запускалось в этой среде; оно должно выполняться в Yandex DataSphere job.
