# Signature Detection: Backend

FastAPI backend for signature detection, with a Celery worker, Redis queue, and PostgreSQL database, all running in Docker.

## Services

| Service    | Purpose                                      |
|------------|----------------------------------------------|
| `api`      | FastAPI app (docs at `/docs`)                |
| `worker`   | Background worker that processes jobs       |
| `redis`    | Message broker / task queue                  |
| `postgres` | Database                                     |

## Requirements

- [Git](https://git-scm.com/)
- [Docker Desktop](https://www.docker.com/products/docker-desktop/) (must be running)

## Run with Docker

### 1. Clone the repository

```bash
git clone https://github.com/AmmardarorgP1/Signature-Detection.git
cd Signature-Detection
```

### 2. Create the `.env` file

Windows (CMD):

```cmd
copy .env.example .env
```

PowerShell:

```powershell
Copy-Item .env.example .env
```

macOS / Linux:

```bash
cp .env.example .env
```

Open `.env` and set `POSTGRES_PASSWORD` to a development password.

> If `.env.example` is missing, create `.env` manually with:
>
> ```
> POSTGRES_PASSWORD=your_password
> ```

### 3. Build and start the backend

```bash
docker compose up -d --build
```

### 4. Check the containers

```bash
docker compose ps
```

The `api`, `worker`, `redis`, and `postgres` services should all be running.

### 5. Open the API docs

http://127.0.0.1:8000/docs

## Useful commands

```bash
# View recent logs
docker compose logs --tail=50 api
docker compose logs --tail=50 worker

# Stop and remove the containers
docker compose down
```

## Notes

- Backend services can run before the trained detection model is integrated.
- Detection jobs will fail until `detection/inference.py` provides `detect_signatures()` and the required model files are added.
- Postgres reads `POSTGRES_PASSWORD` only the first time the database is created. If you change it later, run `docker compose down -v` (this deletes the DB data) and start again.
- Never commit your `.env` file. Only `.env.example` should be in the repo.