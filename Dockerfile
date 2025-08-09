# Dockerfile
FROM python:3.10-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential git curl && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY . .
RUN pip install -e .

CMD ["python", "-m", "faircare.experiments.run_experiments", \
     "--dataset", "heart", "--algo", "faircare", \
     "--num_clients", "3", "--rounds", "3", "--local_epochs", "1", \
     "--batch_size", "64", "--lr", "1e-3", "--lambdaG", "1.0", "--lambdaC", "0.5", "--lambdaA", "0.2", \
     "--q", "0.2", "--beta", "0.8", "--dirichlet_alpha", "0.7", "--sensitive_attr", "sex", \
     "--outdir", "runs/docker_heart/"]
