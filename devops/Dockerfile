FROM python:3.12-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        curl \
        tzdata \
    && rm -rf /var/lib/apt/lists/*

ENV TZ=Europe/Moscow
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

WORKDIR /app

COPY devops/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app ./app
COPY scripts ./scripts
COPY docs ./docs

ENV HF_HOME=/home/appuser/.cache/huggingface

RUN useradd --create-home appuser \
    && mkdir -p /app/logs /home/appuser/.cache/huggingface \
    && chown -R appuser:appuser /app /home/appuser
USER appuser

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]