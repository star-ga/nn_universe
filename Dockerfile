# Reproduction container for the FIM tier-hierarchy experiments.
# Mirrors requirements.txt (the canonical pin file).
#
#   docker build -t nn-universe .
#   docker run --gpus all -it nn-universe bash run_all.sh
#
# CPU-only fallback works for every main-panel experiment; GPU is
# required only for the production-scale checkpoints (see §3.4).
FROM python:3.11-slim

WORKDIR /workspace

RUN apt-get update \
    && apt-get install -y --no-install-recommends git \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt \
    && pip install --no-cache-dir matplotlib pytest

COPY . .

CMD ["bash", "run_all.sh"]
