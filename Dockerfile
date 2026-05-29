#TODO: Pin a specific python:3.11-slim
FROM python:3.11-slim

WORKDIR /app

# System deps required by some sklearn / lightgbm wheels;
RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

CMD ["python", "main.py"]
