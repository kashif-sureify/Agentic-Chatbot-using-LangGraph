FROM python:3.11-slim

# Prevent Python cache files and enable immediate container logs
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    DATA_DIR=/src/app/data

WORKDIR /src/app

# libgomp1 (OpenMP runtime) is needed by faiss-cpu; all other deps ship prebuilt wheels
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for Docker build caching
COPY requirements.txt .

# Install Python dependencies
RUN python -m pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy the complete project
COPY . .

# Chat history (SQLite) and FAISS indexes live here; mount a volume to keep them
VOLUME ["/src/app/data"]

EXPOSE 8501

CMD ["streamlit", "run", "src/app.py", \
     "--server.port=8501", \
     "--server.address=0.0.0.0", \
     "--server.headless=true", \
     "--browser.gatherUsageStats=false"]