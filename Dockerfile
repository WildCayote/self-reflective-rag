# Use an official Python runtime as a parent image
FROM python:3.9-slim

# Set the working directory in the container
WORKDIR /app

# Copy the requirements.txt file into the container
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy all app code into container
COPY . .

# Set working directory where the FastAPI app is
WORKDIR /app/src

# Set Python output to be unbuffered
ENV PYTHONUNBUFFERED=1

# Run the script before launching the server
CMD ["sh", "-c", "python scripts/chat_persistence_service.py && uvicorn app:app --host 0.0.0.0 --port 8000"]
