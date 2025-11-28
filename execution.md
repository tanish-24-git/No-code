How to build & run (dev)

Create .env from the .env.example and modify values.

Build & start:

```
docker compose up --build -d
```

Check logs:

```
docker compose logs -f web
docker compose logs -f redis
```

Enter the container shell (optional):

```
docker compose exec web /bin/sh
```

# or

```
docker exec -it no_code_ml_api /bin/sh
```

Stop & remove:

```
docker compose down
```