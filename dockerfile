FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1
ENV DEBIAN_FRONTEND=noninteractive
ENV VENV_PATH="/opt/venv"

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libglib2.0-0 \
    libsm6 \
    libxrender1 \
    libxext6 \
    libfontconfig1 \
    libice6 \
    ffmpeg \
    python3-pip \
    && rm -rf /var/lib/apt/lists/*

RUN python -m venv /opt/venv

ENV PATH="/opt/venv/bin:$PATH"

COPY requirements.txt .
RUN pip install --no-cache-dir --verbose -r requirements.txt
RUN python3 -m pip install --break-system-packages --verbose "fastapi[standard]"

COPY . /app
WORKDIR /app

EXPOSE 6969

CMD ["fastapi", "run", "--port", "6969"]
