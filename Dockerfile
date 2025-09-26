# CPU base image with Python
FROM python:3.10-slim

# System deps for librosa and audio (ffmpeg)
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Create workdir
WORKDIR /app

# Install build deps
RUN pip install --no-cache-dir --upgrade pip

# Copy project and install
COPY pyproject.toml README.md /app/
COPY src /app/src

RUN pip install --no-cache-dir .

# Default command: provide CLI usage help
CMD ["naijaligo-asr", "--help"]
