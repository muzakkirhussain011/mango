# Research-grade Dockerfile for FairCare-FL
FROM pytorch/pytorch:2.3.1-cuda12.1-cudnn8-runtime

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y git build-essential libopenmpi-dev && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace/mango
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
RUN pip install -e .

CMD ["python", "-m", "faircare.experiments.run_experiments", "--config", "faircare/experiments/configs/default.yaml"]
