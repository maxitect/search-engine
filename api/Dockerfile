FROM python:3.13-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the application code
COPY . .

# Entry point script
COPY entrypoint.sh /
RUN chmod +x /entrypoint.sh

# Expose the port for FastAPI
EXPOSE 8000

# Set the entrypoint
ENTRYPOINT ["/entrypoint.sh"]