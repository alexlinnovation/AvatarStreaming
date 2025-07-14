FROM nvcr.io/nvidia/tensorrt:23.12-py3

# ───── SYSTEM DEPENDENCIES ────────────────────────────────────
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    ffmpeg \
    libsndfile1 \
    libturbojpeg0-dev \
    libsm6 libxext6 libgl1 \
    git curl wget \
    && rm -rf /var/lib/apt/lists/*

# ───── SET WORKDIR AND COPY DEPENDENCIES FIRST ────────────────
WORKDIR /app

# Copy just requirements first to cache pip install
COPY requirements.txt ./
# RUN pip install --upgrade pip
RUN pip install -r requirements.txt
RUN pip install numpy==2.0.1
RUN pip install kokoro-onnx==0.3.9
RUN pip install onnxruntime-gpu==1.22.0

# ───── NOW COPY THE REST OF THE SOURCE ────────────────────────
COPY . .

# ───── EXPOSE PORT + RUN ──────────────────────────────────────
EXPOSE 8010
CMD ["uvicorn", "online:app", "--host", "0.0.0.0", "--port", "8010", "--reload"]
