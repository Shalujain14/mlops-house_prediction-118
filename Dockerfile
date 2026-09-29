FROM python:3.12-slim

WORKDIR /app

# Install Python dependencies
COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY serve.py .

# Copy frontend files
COPY static ./static

# FastAPI port
EXPOSE 8000

TRAINING_URI http://100.59.112.41:5000

# Start FastAPI
CMD ["uvicorn", "serve:app", "--host", "0.0.0.0", "--port", "8000"]

