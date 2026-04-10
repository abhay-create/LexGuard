FROM python:3.10-slim-bookworm

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends ca-certificates curl build-essential && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN python -m pip install --no-cache-dir -r requirements.txt
RUN python -m pip install --no-cache-dir torch==2.2.0+cu118 torchvision==0.17.0+cu118 torchaudio==2.2.0+cu118 --index-url https://download.pytorch.org/whl/cu118
RUN curl -fsSL https://ollama.com/install.sh | sh
COPY . .

CMD [ "/bin/bash" ]