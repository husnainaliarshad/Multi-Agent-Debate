# --- Stage 1: Builder ---
FROM python:3.11-slim AS builder
WORKDIR /app

# Install build tools
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Create a virtual environment
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Install everything INTO the virtual environment
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# --- Stage 2: Final Runtime ---
FROM python:3.11-slim
WORKDIR /app

# Install runtime-only system dependencies
RUN apt-get update && apt-get install -y curl && rm -rf /var/lib/apt/lists/*

# Copy the entire virtual environment
COPY --from=builder /opt/venv /opt/venv
COPY . .

# Use the virtual environment
ENV PATH="/opt/venv/bin:$PATH"

EXPOSE 8000
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
