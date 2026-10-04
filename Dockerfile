# Imagem para Cloud Run. Build: gcloud run deploy --source . (Cloud Build).
FROM python:3.12-slim-bookworm

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PLAYWRIGHT_BROWSERS_PATH=/ms-playwright

RUN apt-get update \
    && apt-get install -y --no-install-recommends ffmpeg \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .
# --only-shell: headless=True usa o chromium-headless-shell (imagem menor).
RUN pip install -r requirements.txt \
    && playwright install --with-deps --only-shell chromium \
    && rm -rf /var/lib/apt/lists/*

COPY scriptmax ./scriptmax
COPY static ./static
COPY fonts ./fonts

RUN useradd --create-home --uid 1000 app
USER app

# HOST 0.0.0.0 exige APP_TOKEN (validado em config.py). PORT é injetado pelo Cloud Run.
# DATA_DIR aponta para o bucket montado como volume (relatórios e biblioteca persistem).
ENV HOST=0.0.0.0 \
    BEHIND_PROXY=1 \
    DATA_DIR=/data

CMD ["python", "-m", "scriptmax"]
