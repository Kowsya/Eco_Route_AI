# ---------------------------------------------------------------
# EcoRoute AI - container image for Cloud Run
# Each instruction creates a cached layer; order = least -> most frequently changed.
# ---------------------------------------------------------------

# 1) Base image: small official Python (must match the Python used for training: 3.11)
FROM python:3.11-slim

# 2) Python settings: no .pyc files, unbuffered logs (so logs reach Cloud Logging instantly)
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# 3) Work inside /app
WORKDIR /app

# 4) Install dependencies FIRST (this layer is cached unless requirements change)
COPY requirements-api.txt .
RUN pip install --no-cache-dir -r requirements-api.txt

# 5) Copy only what the API needs: code + trained models + frontend (NOT the dataset)
COPY src/ ./src/
COPY api/ ./api/
COPY models/ ./models/
COPY frontend/ ./frontend/

# 6) Security: run as a normal user instead of root
RUN useradd --create-home appuser
USER appuser

# 7) Cloud Run tells the container which port to use through the PORT variable
ENV PORT=8080
EXPOSE 8080

# 8) Start the web server. 'sh -c' lets us use the $PORT variable.
CMD ["sh", "-c", "uvicorn api.main:app --host 0.0.0.0 --port ${PORT}"]