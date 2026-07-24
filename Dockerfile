FROM python:3.14-slim

WORKDIR /app

RUN apt-get update && apt-get install -y \
    build-essential \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml uv.lock /app/
COPY . /app

RUN pip install uv && uv sync --frozen --group full

ENV PYTHONUNBUFFERED=1

CMD ["eqprop-verify", "--quick"]
