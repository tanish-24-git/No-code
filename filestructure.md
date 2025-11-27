```
no-code-ml/
├─ Dockerfile
├─ docker-compose.yml
├─ requirements.txt
├─ .dockerignore
├─ .env.example
├─ README.md
├─ main.py
├─ config/
│  └─ settings.py
├─ app/
│  └─ redis_client.py
├─ models/
│  ├─ requests.py
│  └─ responses.py
├─ services/
│  ├─ file_service.py
│  ├─ insight_service.py
│  ├─ llm_service.py
│  ├─ ml_service.py
│  └─ preprocessing_service.py
├─ utils/
│  ├─ exceptions.py
│  └─ validators.py
└─ uploads/   # mounted volume (created by docker-compose)
```
