# Open-LLM-webchat

FROM python:3.12-slim

# Basic environment configuration (logs, timezone and npm cache)
ENV PYTHONUNBUFFERED=1 \
    TZ=America/Sao_Paulo \
    NPM_CONFIG_CACHE=/tmp/.npm

# Set working directory
WORKDIR /home/app/src

# Install system dependencies, Docker CLI, and Node.js
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        curl ca-certificates gnupg libgssapi-krb5-2 tzdata && \
    cp /usr/share/zoneinfo/${TZ} /etc/localtime && echo ${TZ} > /etc/timezone && \
    install -m 0755 -d /etc/apt/keyrings && \
    curl -fsSL https://download.docker.com/linux/debian/gpg | tee /etc/apt/keyrings/docker.asc > /dev/null && \
    chmod a+r /etc/apt/keyrings/docker.asc && \
    echo "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.asc] https://download.docker.com/linux/debian bookworm stable" > /etc/apt/sources.list.d/docker.list && \
    apt-get update && apt-get install -y docker-cli && \
    curl -fsSL https://deb.nodesource.com/setup_20.x | bash - && \
    apt-get install -y nodejs && \
    npm install -g npm@11.6.2 && \
    rm -rf /var/lib/apt/lists/*

# Create non-root user
RUN groupadd -r appgroup && \
    useradd --no-log-init -r -g appgroup -m -d /home/app/appuser appuser && \
    chown -R appuser:appgroup /home/app/src

# Ensure appuser has read/write permissions on temporary directories
RUN mkdir -p /home/app/src/tmp && \
    mkdir -p /tmp/.npm && \
    chown -R appuser:appgroup /home/app/src/tmp && \
    chown -R appuser:appgroup /tmp/.npm 

# Copy requirements and install Python deps
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip wheel && \
    pip install --no-cache-dir -r requirements.txt

# Copy all codes
COPY src/ .

# Set the default user 
USER appuser

# Initial command for the container
CMD ["python", "app.py"]
