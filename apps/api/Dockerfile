FROM python:3.11-slim

WORKDIR /app

# Install backend dependencies
COPY apps/api/requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Install shared recsys_core package
COPY libs/recsys_core /app/libs/recsys_core
RUN pip install --no-cache-dir -e /app/libs/recsys_core

# Copy API source code
COPY apps/api /app

EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
