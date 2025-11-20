# syntax=docker/dockerfile:1.5

# ============================================================================
# Build Stage
# ============================================================================
FROM python:3.11-slim AS builder

WORKDIR /app

# Install build dependencies for TA-Lib
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        gcc \
        g++ \
        make \
        wget \
        ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Install TA-Lib C library
RUN wget http://prdownloads.sourceforge.net/ta-lib/ta-lib-0.4.0-src.tar.gz && \
    tar -xzf ta-lib-0.4.0-src.tar.gz && \
    cd ta-lib/ && \
    ./configure --prefix=/usr && \
    make && \
    make install && \
    cd .. && \
    rm -rf ta-lib ta-lib-0.4.0-src.tar.gz

# Copy dependency files
COPY pyproject.toml ./

# Install Poetry and dependencies
RUN pip install --no-cache-dir poetry==1.7.1 && \
    poetry config virtualenvs.create false && \
    poetry install --only main --no-interaction --no-ansi

# ============================================================================
# Runtime Stage
# ============================================================================
FROM python:3.11-slim

WORKDIR /app

# Install runtime dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        curl \
        ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Copy TA-Lib library from builder
COPY --from=builder /usr/lib/libta_lib.* /usr/lib/

# Create non-root user
RUN useradd -m -u 1000 appuser && \
    chown -R appuser:appuser /app

# Copy Python packages from builder
COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# Copy application code
COPY --chown=appuser:appuser app ./app
COPY --chown=appuser:appuser pyproject.toml ./

# Environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONPATH=/app \
    PORT=8000

# Switch to non-root user
USER appuser

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=3s --start-period=40s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Start command
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]