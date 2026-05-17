# Qwen3-VL-8B-Instruct: SFT + GRPO на `top-papers-graph-experts-data/exports/colab-run-001` в Yandex DataSphere Jobs

Дата ревизии: 2026-05-17.

Этот tutorial описывает запуск полного дообучения VLM-модели `Qwen/Qwen3-VL-8B-Instruct` на подготовленном Hugging Face export `top-papers/top-papers-graph-experts-data/exports/colab-run-001`. Конфигурация рассчитана на Yandex DataSphere `g2.2` и общий бюджет проекта **100 000 ₽**. Настройки подобраны так, чтобы использовать почти весь бюджет на качество, но оставить резерв и не превысить лимит.

## 1. Что именно было исправлено и почему

Главная проблема старой версии пайплайна: сборщик датасета читал общий HF dataset как `imagefolder`/classification split и создавал простые `image -> label` задачи. Для указанного export это неверно, потому что в `exports/colab-run-001` уже лежат готовые TRL-совместимые файлы:

- `sft.jsonl` — SFT-примеры с `messages`, `images`, `task_family`, `metadata`;
- `grpo.jsonl` — GRPO-примеры с `prompt`, `images`, `reference_json`, `reference_assertions_json`, `expected_verdict` и другими полями для reward-функций.

Теперь сборщик по умолчанию работает в режиме:

```yaml
HF_DATASET_SOURCE_MODE: export
HF_DATASET_EXPORT_SUBDIR: exports/colab-run-001
```

и строит локальные файлы:

```text
data/derived/hf_top_papers_graph_experts/sft_all.jsonl
data/derived/hf_top_papers_graph_experts/sft_train.jsonl
data/derived/hf_top_papers_graph_experts/sft_eval.jsonl
data/derived/hf_top_papers_graph_experts/grpo_all.jsonl
data/derived/hf_top_papers_graph_experts/grpo_train.jsonl
data/derived/hf_top_papers_graph_experts/grpo_eval.jsonl
data/derived/hf_top_papers_graph_experts/summary.json
```

## 2. Основные файлы в репозитории

Основной DataSphere job:

```text
experiments/vlm_finetuning/datasphere/job_configs/hf_top_papers_sft_grpo_full_g2_2.yaml
```

Runtime wrapper внутри job:

```text
experiments/vlm_finetuning/datasphere/bin/run_hf_top_papers_sft_grpo_full.sh
```

Сборка HF export в локальные train/eval JSONL:

```text
experiments/vlm_finetuning/scripts/build_hf_graph_experts_dataset.py
```

SFT entrypoint:

```text
experiments/vlm_finetuning/scripts/train_vlm_sft.py
```

GRPO entrypoint:

```text
experiments/vlm_finetuning/scripts/train_vlm_grpo.py
```

Оценка стоимости:

```text
experiments/vlm_finetuning/scripts/estimate_datasphere_costs.py
```

Managed launcher:

```text
experiments/vlm_finetuning/datasphere/run_full_pipeline.py
experiments/vlm_finetuning/datasphere/launch_examples.sh
```

## 3. Итоговая аппаратная конфигурация

Используется DataSphere `g2.2`:

```yaml
cloud-instance-types:
  - g2.2

working-storage:
  type: SSD
  size: 1024Gb
```

Почему `g2.2`, а не `g2.1` или `g2.4`:

- `g2.1` дешевле, но одна A100/80 GB менее удобна для Qwen3-VL-8B + multi-image GRPO;
- `g2.4` быстрее, но стоит ровно в 2 раза дороже `g2.2`, поэтому при фиксированном бюджете сокращает wall-clock бюджет и число итераций;
- `g2.2` даёт 2 A100 и 160 GB VRAM суммарно, что позволяет держать LoRA rank 32, multi-image VLM inputs и GRPO с `num_generations=2`.

## 4. Бюджетный расчёт

В конфиге зафиксировано:

```yaml
BUDGET_RUB: 100000
BUDGET_RESERVE_RUB: 5000
BUDGET_SHUTDOWN_MARGIN_SECONDS: 900
G2_2_RUB_PER_HOUR: 1085.76
DATA_TIMEOUT_HOURS: 3
SFT_TIMEOUT_HOURS: 48
GRPO_TIMEOUT_HOURS: 35
HF_UPLOAD_TIMEOUT_HOURS: 1
```

Суммарные phase timeouts:

```text
3 + 48 + 35 + 1 = 87 часов
```

Worst-case compute cost:

```text
87 * 1085.76 = 94 461.12 ₽
```

С учётом резерва:

```text
94 461.12 + 5 000 = 99 461.12 ₽
```

Остаток до лимита:

```text
100 000 - 99 461.12 = 538.88 ₽
```

В wrapper также встроен глобальный wall-clock guard:

```text
MAX_COMPUTE_SECONDS = ((BUDGET_RUB - BUDGET_RESERVE_RUB) / G2_2_RUB_PER_HOUR * 3600) - 900
```

То есть job не просто полагается на отдельные `timeout`, а проверяет общий бюджетный deadline перед каждой стадией. Это не заменяет официальный бюджет/квоты в Yandex Cloud, но защищает сам pipeline от неконтролируемого перерасхода в рамках заданных переменных.

Проверить расчёт можно локально:

```bash
python experiments/vlm_finetuning/scripts/estimate_datasphere_costs.py \
  --scenario qwen3vl_8b_sft_grpo_full_budget_guarded \
  --out reports/cost_estimate_qwen3vl_budget.md \
  --format markdown
```

## 5. Почему выбраны такие training settings

### SFT

```yaml
MAX_SFT_STEPS: 480
SFT_EPOCHS: 4
SFT_LR: 7e-5
SFT_WARMUP_RATIO: 0.05
SFT_LR_SCHEDULER: cosine
SFT_WEIGHT_DECAY: 0.01
SFT_MAX_GRAD_NORM: 0.3
SFT_LORA_R: 32
SFT_LORA_ALPHA: 64
SFT_LORA_DROPOUT: 0.05
SFT_PER_DEVICE_BATCH: 1
SFT_GRAD_ACCUM: 8
SFT_SAVE_STEPS: 60
SFT_EVAL_STEPS: 60
```

Ключевые моменты:

- `assistant_only_loss=True`, чтобы SFT оптимизировал ответы ассистента, а не системный/user контекст;
- `max_length=None`, чтобы не обрезать image tokens в VLM-примерах;
- LoRA `r=32`, `alpha=64` — больше адаптационной ёмкости, чем `r=16`, но всё ещё безопасно для `g2.2`;
- `gradient_checkpointing=True`, `bf16=True`, `tf32=True` для памяти и скорости;
- `ATTN_IMPLEMENTATION=auto`: если `flash_attn` установлен, будет использован `flash_attention_2`, иначе безопасный fallback на `sdpa`.

### GRPO

```yaml
MAX_GRPO_STEPS: 160
GRPO_EPOCHS: 1
GRPO_LR: 1e-5
GRPO_WARMUP_RATIO: 0.08
GRPO_PER_DEVICE_BATCH: 1
GRPO_GRAD_ACCUM: 8
GRPO_NUM_GENERATIONS: 2
GRPO_NUM_GENERATIONS_EVAL: 2
GRPO_MAX_COMPLETION_LENGTH: 384
GRPO_TEMPERATURE: 0.8
GRPO_TOP_P: 0.95
GRPO_IMPORTANCE_SAMPLING_LEVEL: sequence
GRPO_MULTI_OBJECTIVE_AGGREGATION: normalize_then_sum
GRPO_REWARD_WEIGHTS: "0.0 1.0 0.8 1.2 0.5 1.5"
```

На `g2.2` эффективный global batch для GRPO:

```text
2 GPU * 1 per_device_batch * 8 grad_accum = 16
```

Он делится на `num_generations=2`, что важно для GRPO. `max_completion_length=384` оставляет место для structured JSON/reasoning output, но не раздувает стоимость генерации до небезопасного уровня.

Reward weights соответствуют шести reward-функциям в `train_vlm_grpo.py`:

```text
label_format: 0.0
schema_json: 1.0
temporal_fields: 0.8
graph_fields: 1.2
evidence_grounding: 0.5
verdict_exact_match: 1.5
```

`label_format` оставлен с нулевым весом, потому что основной export не является простой classification задачей. Остальные веса усиливают JSON-схему, графовую консистентность и совпадение экспертного verdict.

## 6. VLM preprocessing guardrails

```yaml
MAX_IMAGES_PER_EXAMPLE_SFT: 3
MAX_IMAGES_PER_EXAMPLE_GRPO: 2
VLM_MAX_PIXELS: 1003520
```

Причина:

- в SFT можно сохранить больше визуального контекста;
- в GRPO стоимость растёт быстрее, потому что модель генерирует несколько completions на prompt;
- `VLM_MAX_PIXELS=1003520` ограничивает размер изображения примерно до уровня около 1 Мп, что снижает риск CUDA OOM и ускоряет attention.

Если будет OOM, сначала уменьшайте именно эти параметры:

```yaml
VLM_MAX_PIXELS: 802816
MAX_IMAGES_PER_EXAMPLE_SFT: 2
MAX_IMAGES_PER_EXAMPLE_GRPO: 1
GRPO_MAX_COMPLETION_LENGTH: 256
GRPO_GRAD_ACCUM: 4
MAX_GRPO_STEPS: 80
```

## 7. Подготовка локального окружения

Из корня репозитория:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -U pip
python -m pip install -U datasphere pyyaml huggingface_hub

datasphere version
```

Если используете Yandex Cloud CLI:

```bash
yc init
datasphere project list -c <community_id>
```

Если используете OAuth token:

```bash
export YC_OAUTH_TOKEN='<oauth_token>'
datasphere -t "$YC_OAUTH_TOKEN" project list -c <community_id>
```

Задайте project id:

```bash
export DATASPHERE_PROJECT_ID='<project_id>'
datasphere project get --id "$DATASPHERE_PROJECT_ID"
```

## 8. Hugging Face token

Если включена автозагрузка результата:

```yaml
HF_UPLOAD_AFTER_TRAINING: "1"
HF_REPO_ID: top-papers/Qwen3-VL-8B-Instruct-scireason
```

то внутри DataSphere job должен быть доступен токен:

```text
HF_TOKEN
```

или

```text
HUGGING_FACE_HUB_TOKEN
```

Рекомендуется добавить его как DataSphere secret, а не прописывать в YAML.

Локальная проверка токена:

```bash
export HF_TOKEN='<hf_write_token>'
python - <<'PY'
from huggingface_hub import HfApi
print(HfApi(token=None).whoami())
PY
```

Если upload не нужен, перед запуском временно поставьте:

```yaml
HF_UPLOAD_AFTER_TRAINING: "0"
```

## 9. Preflight-проверки

Эти проверки не запускают GPU training:

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
```

Проверка YAML configs:

```bash
python - <<'PY'
from pathlib import Path
import yaml
for path in sorted(Path('experiments/vlm_finetuning/datasphere/job_configs').glob('*.yaml')):
    cfg = yaml.safe_load(path.read_text(encoding='utf-8'))
    py = cfg.get('env', {}).get('python', {})
    assert not ('root-path' in py and 'local-paths' in py), path
    storage = cfg.get('working-storage') or {}
    assert storage.get('type') == 'SSD', path
    assert str(storage.get('size', '')).endswith('Gb'), path
    print('[OK]', path)
PY
```

Быстрые тесты:

```bash
python -m pytest -q tests/test_datasphere_job_configs.py tests/test_vlm_training_format_normalization.py
```

Dry-run managed launcher:

```bash
python experiments/vlm_finetuning/datasphere/run_full_pipeline.py \
  --project-id "$DATASPHERE_PROJECT_ID" \
  --dry-run
```

## 10. Запуск полного эксперимента

Рекомендуемый вариант:

```bash
export DATASPHERE_PROJECT_ID='<project_id>'
bash experiments/vlm_finetuning/datasphere/launch_examples.sh hf-full-managed
```

Прямой запуск через CLI:

```bash
datasphere project job execute \
  -p "$DATASPHERE_PROJECT_ID" \
  -c experiments/vlm_finetuning/datasphere/job_configs/hf_top_papers_sft_grpo_full_g2_2.yaml
```

Managed launcher после завершения пытается выполнить:

```bash
datasphere project job set-data-ttl --id <job_id> --days 1
datasphere project job download-files --id <job_id>
```

TTL в 1 день важен: DataSphere хранит input cache, environments, logs и outputs, и это хранение тарифицируется отдельно. Для полного 1 TB рабочего объёма даже несколько дней хранения могут заметно съесть резерв.

## 11. Мониторинг job

Список jobs:

```bash
bash experiments/vlm_finetuning/datasphere/launch_examples.sh list
```

Информация по job:

```bash
bash experiments/vlm_finetuning/datasphere/launch_examples.sh get <job_id>
```

Подключиться к логам:

```bash
bash experiments/vlm_finetuning/datasphere/launch_examples.sh attach <job_id>
```

Остановить job:

```bash
bash experiments/vlm_finetuning/datasphere/launch_examples.sh cancel <job_id>
```

Скачать outputs вручную:

```bash
bash experiments/vlm_finetuning/datasphere/launch_examples.sh download <job_id>
```

Сократить TTL вручную:

```bash
bash experiments/vlm_finetuning/datasphere/launch_examples.sh ttl <job_id> 1
```

## 12. Ожидаемые результаты

После успешного запуска:

```text
outputs/hf_top_papers_qwen3vl_8b_sft_lora.tar.gz
outputs/hf_top_papers_qwen3vl_8b_grpo_lora.tar.gz
reports/hf_top_papers_qwen3vl_8b_datasphere_reports.tar.gz
reports/hf_top_papers_qwen3vl_8b_datasphere/budget_plan.json
reports/hf_top_papers_qwen3vl_8b_datasphere/final_summary.json
reports/hf_top_papers_qwen3vl_8b_datasphere/hf_upload_summary.json
reports/hf_top_papers_qwen3vl_8b_datasphere/hf_upload_bundle/artifacts/reports/hf_upload_manifest.json
```

Проверка:

```bash
ls -lh outputs/*hf_top_papers*qwen3vl*tar.gz
python -m json.tool data/derived/hf_top_papers_graph_experts/summary.json | head -120
python -m json.tool reports/hf_top_papers_qwen3vl_8b_datasphere/budget_plan.json | head -120
python -m json.tool reports/hf_top_papers_qwen3vl_8b_datasphere/final_summary.json | head -120
```

Проверка JSONL:

```bash
python - <<'PY'
import json
from pathlib import Path
for path in [
    Path('data/derived/hf_top_papers_graph_experts/sft_train.jsonl'),
    Path('data/derived/hf_top_papers_graph_experts/grpo_train.jsonl'),
]:
    print('\n---', path)
    with path.open(encoding='utf-8') as f:
        row = json.loads(next(f))
    print(json.dumps({
        'id': row.get('id'),
        'task_family': row.get('task_family'),
        'images_count': len(row.get('images') or []),
        'has_messages': 'messages' in row,
        'has_prompt': 'prompt' in row,
        'has_reference_json': 'reference_json' in row,
        'has_expected_verdict': 'expected_verdict' in row,
    }, ensure_ascii=False, indent=2))
PY
```

## 13. Как безопасно менять настройки

### Увеличить качество без выхода за бюджет

Сначала смотрите фактическую скорость первых 30-60 минут job. Если видно, что пайплайн сильно не доиспользует бюджет, увеличивайте только один параметр за раз:

```yaml
MAX_SFT_STEPS: 560
# или
MAX_GRPO_STEPS: 200
```

После любого увеличения пересчитайте budget:

```bash
python experiments/vlm_finetuning/scripts/estimate_datasphere_costs.py \
  --hours <new_total_hours> \
  --out reports/cost_estimate_custom.md \
  --format markdown
```

Не увеличивайте сумму `DATA_TIMEOUT_HOURS + SFT_TIMEOUT_HOURS + GRPO_TIMEOUT_HOURS + HF_UPLOAD_TIMEOUT_HOURS` выше 87 часов без уменьшения `BUDGET_RESERVE_RUB` или официального увеличения бюджета.

### Уменьшить риск OOM

```yaml
VLM_MAX_PIXELS: 802816
MAX_IMAGES_PER_EXAMPLE_GRPO: 1
GRPO_MAX_COMPLETION_LENGTH: 256
GRPO_NUM_GENERATIONS: 2
GRPO_GRAD_ACCUM: 4
```

### Smoke run

Для проверки окружения перед дорогим запуском:

```yaml
MAX_SFT_STEPS: 40
MAX_GRPO_STEPS: 10
DATA_TIMEOUT_HOURS: 1
SFT_TIMEOUT_HOURS: 4
GRPO_TIMEOUT_HOURS: 4
HF_UPLOAD_AFTER_TRAINING: "0"
```

## 14. Troubleshooting

### `HF upload is enabled, but HF_TOKEN/HUGGING_FACE_HUB_TOKEN is not set`

Либо добавьте `HF_TOKEN` как DataSphere secret, либо временно выключите upload:

```yaml
HF_UPLOAD_AFTER_TRAINING: "0"
```

### CUDA OOM в SFT

Уменьшите:

```yaml
VLM_MAX_PIXELS: 802816
MAX_IMAGES_PER_EXAMPLE_SFT: 2
SFT_GRAD_ACCUM: 4
```

### CUDA OOM в GRPO

Уменьшите:

```yaml
MAX_IMAGES_PER_EXAMPLE_GRPO: 1
GRPO_MAX_COMPLETION_LENGTH: 256
GRPO_GRAD_ACCUM: 4
MAX_GRPO_STEPS: 80
```

### `num_generations` несовместим с batch size

Для `g2.2` текущий расчёт:

```text
2 GPU * 1 batch * 8 grad_accum = 16
```

`16` делится на `GRPO_NUM_GENERATIONS=2`. Если меняете `GRPO_GRAD_ACCUM`, проверьте, что новый effective batch всё ещё делится на `GRPO_NUM_GENERATIONS`.

### DataSphere job не видит файлы

Запускайте команды из корня репозитория. В job config в `local-paths` передаются:

```yaml
- experiments/vlm_finetuning/
- data/
- examples/
```

### Накопились расходы на хранение job data

Сразу после завершения:

```bash
bash experiments/vlm_finetuning/datasphere/launch_examples.sh ttl <job_id> 1
```

После скачивания outputs можно очистить cache/job data в UI DataSphere Jobs.

## 15. Источники для сверки

- Qwen3-VL-8B-Instruct model card: https://huggingface.co/Qwen/Qwen3-VL-8B-Instruct
- HF dataset export: https://huggingface.co/datasets/top-papers/top-papers-graph-experts-data/tree/main/exports/colab-run-001
- HF export README: https://huggingface.co/datasets/top-papers/top-papers-graph-experts-data/raw/main/exports/colab-run-001/README.md
- Yandex DataSphere Jobs: https://yandex.cloud/en/docs/datasphere/concepts/jobs/
- Yandex DataSphere configurations: https://yandex.cloud/ru/docs/datasphere/concepts/configurations
- Yandex DataSphere pricing: https://yandex.cloud/ru/docs/datasphere/pricing
- Yandex Compute GPU concepts: https://yandex.cloud/ru/docs/compute/concepts/gpus
- TRL SFTTrainer: https://huggingface.co/docs/trl/sft_trainer
- TRL GRPOTrainer: https://huggingface.co/docs/trl/grpo_trainer
- PyPI TRL: https://pypi.org/project/trl/
