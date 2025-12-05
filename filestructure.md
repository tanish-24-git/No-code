```
Directory structure:
└── tanish-24-git-temp-nocode/
    ├── docker-compose.yml
    ├── Dockerfile
    ├── execution.md
    ├── filestructure.md
    ├── main.py
    ├── preprocessing_plan.md
    ├── requirements.txt
    ├── .dockerignore
    ├── .env.md
    ├── app/
    │   └── redis_client.py
    ├── config/
    │   └── settings.py
    ├── models/
    │   ├── requests.py
    │   └── responses.py
    ├── scripts/
    │   └── download_models.py
    ├── services/
    │   ├── file_service.py
    │   ├── insight_service.py
    │   ├── llm_service.py
    │   ├── ml_service.py
    │   ├── preprocessing_engine.py
    │   ├── preprocessing_service.py
    │   └── ops/
    │       ├── nlp/
    │       │   ├── __init__.py
    │       │   ├── chunking_ops.py
    │       │   ├── cleaning_ops.py
    │       │   ├── dedup_ops.py
    │       │   ├── jsonl_ops.py
    │       │   ├── metadata_ops.py
    │       │   ├── pii_ops.py
    │       │   ├── prompt_ops.py
    │       │   ├── splitting_ops.py
    │       │   └── tokenization_ops.py
    │       ├── shared/
    │       │   ├── __init__.py
    │       │   ├── artifact_ops.py
    │       │   ├── parsing_ops.py
    │       │   └── safeguards_ops.py
    │       └── tabular/
    │           ├── __init__.py
    │           ├── cleaning_ops.py
    │           ├── encoding_ops.py
    │           ├── fe_ops.py
    │           ├── missing_ops.py
    │           ├── row_ops.py
    │           ├── scaling_ops.py
    │           ├── selection_ops.py
    │           └── type_fix_ops.py
    ├── uploads/
    │   └── trained_model_preprocessed_9c6c2eb5bae64bc89f413f9033c38ff3.pkl
    └── utils/
        ├── exceptions.py
        └── validators.py

```
