# Обновление блокнотов: README и источники изображений

В архиве обновлены два Colab-блокнота:

1. `top_papers_graph_scidatapipe_hf_colab_from_csv_only_fixed_gdown_scope_assets_fixed.ipynb`
2. `notebooks/top_papers_graph_task3_hf_benchmark_colab.ipynb`

## Что добавлено

В оба блокнота добавлен отдельный шаг перед загрузкой в Hugging Face:

- создание подробного `README.md` на русском языке;
- создание `ARTICLE_IMAGE_SOURCES.md` со ссылками на статьи, из которых взяты изображения/отрендеренные PDF-страницы;
- создание `article_image_sources.jsonl` как машинно-читаемой версии файла источников.

## Проверка Task 1 YAML с длинным именем

Проверен файл вида:

`timofeev_kirill_anatolevich__a63049dbf8f0 - Кирилл Тимофеев.yaml`

Файл успешно читается как YAML и нормализуется Task 1 normalizer без ошибки имени файла. Нормализованный `submission_id` сохраняет канонический идентификатор и русскоязычный suffix автора.
