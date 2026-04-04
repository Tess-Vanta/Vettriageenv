FROM python:3.11-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy environment code
COPY vettriagevenv/ ./vettriagevenv/
COPY server/ ./server/
COPY openenv.yaml .
COPY baseline.py .
COPY README.md .

# Expose port for HF Spaces
EXPOSE 7860

# Start the FastAPI server
ENV ENABLE_WEB_INTERFACE=true
CMD ["uvicorn", "server.app:app", "--host", "0.0.0.0", "--port", "7860"]
