# Path A (default): CPU-only reproduction from shipped results (Path B needs an nvidia/cuda base).
FROM python:3.12-slim
WORKDIR /artifact
RUN apt-get update && apt-get install -y --no-install-recommends xz-utils \
    && rm -rf /var/lib/apt/lists/*
COPY . .
RUN pip install --no-cache-dir ".[reproduce]"
CMD ["./reproduce.sh"]
