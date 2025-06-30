FROM python:3.10-slim

# ───── SYSTEM DEPENDENCIES ────────────────────────────────────
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    ffmpeg \
    libsndfile1 \
    libturbojpeg0-dev \
    libsm6 libxext6 libgl1 \
    git curl wget \
    && rm -rf /var/lib/apt/lists/*

# ───── SET WORKDIR AND COPY FILES ─────────────────────────────
WORKDIR /app
COPY . /app

# ───── INSTALL PYTHON DEPENDENCIES ────────────────────────────
RUN pip install --upgrade pip
RUN pip install numpy==2.0.1
RUN pip install -r requirements.txt

# ───── EXPOSE PORT + RUN ──────────────────────────────────────
EXPOSE 8010
CMD uvicorn online:app --host localhost --port 8010 --reload
