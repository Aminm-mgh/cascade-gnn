FROM python:3.11-slim

WORKDIR /app

# Install dependencies first (better layer caching — only reinstalls
# if requirements.txt actually changes, not on every code edit)
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY src/ ./src/

# data/processed/graph.pt and gcn_weights.pt are NOT copied here —
# they're generated artifacts (91MB+ raw dataset dependency, gitignored).
# Mount them at runtime: docker run -v $(pwd)/data:/app/data ...

EXPOSE 8000

CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]