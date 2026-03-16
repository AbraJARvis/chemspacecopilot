# Installation

## Prerequisites

- Python 3.11
- [uv](https://docs.astral.sh/uv/) package manager
- A [DeepSeek](https://platform.deepseek.com/) API key for the default cloud backend, or a local [Ollama](https://ollama.com/) instance

## Install Dependencies

```bash
uv sync
```

## Environment Configuration

For file-based configuration, copy `.env.example` to `.env` in the project root:

```bash
# Required only for the default DeepSeek provider
DEEPSEEK_API_KEY=your-api-key-here

# Optional model overrides (otherwise .modelconf is used)
# MODEL_PROVIDER=deepseek
# MODEL_ID=deepseek-chat
# OLLAMA_HOST=http://localhost:11434

# Optional — S3/MinIO storage (disable with USE_S3=false)
USE_S3=true
S3_ENDPOINT_URL=http://localhost:9000
MINIO_ACCESS_KEY=cs_copilot
MINIO_SECRET_KEY=chempwd123
ASSETS_BUCKET=chatbot-assets
```

The repository also includes a tracked `.modelconf` file. Edit it if you want to switch from the default DeepSeek backend to a local Ollama model.

## Running the Application

### Chainlit App

```bash
uv run chainlit run chainlit_app.py -w
```

Access the application at **http://localhost:8000**.

Notes:

- The bundled `chainlit.toml` currently has `[persistence] enabled = false`.
- The app sets a per-thread title from your first message; you can rename it in the UI.

### Jupyter Notebook

An example workflow is available in `notebooks/cs_copilot.ipynb`.

## Optional Services

### S3/MinIO Storage

```bash
# Run the interactive setup script
python scripts/setup_s3.py

# Or start MinIO manually
docker run -d --name minio \
  -p 9000:9000 -p 9001:9001 \
  -v /mnt/data:/data \
  -e MINIO_ROOT_USER=cs_copilot \
  -e MINIO_ROOT_PASSWORD=chempwd123 \
  minio/minio server /data --console-address ":9001"
```

If the container already exists: `docker start minio`

### Optional Chainlit Persistence

Chainlit persistence is disabled by default in `chainlit.toml`. Only set up PostgreSQL if you plan to enable Chainlit persistence manually.

```bash
docker run --name chainlit-pg -p 5432:5432 -d \
  -e POSTGRES_PASSWORD=postgres \
  -e POSTGRES_USER=postgres \
  -e POSTGRES_DB=chainlit \
  postgres:16

export DATABASE_URL="postgresql://postgres:postgres@localhost:5432/chainlit"
```

If the container already exists: `docker start chainlit-pg`
