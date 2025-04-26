# Use the official lightweight Python image.
FROM python:3.11-slim

# Prevents Python from buffering stdout and stderr
ENV PYTHONUNBUFFERED True

# Set the working directory in the container
ENV APP_HOME /app
WORKDIR $APP_HOME

# Copy the entire project context
COPY . ./

# Install dependencies (ensure uvicorn is in requirements.txt, gunicorn is now optional)
# You can remove gunicorn from requirements.txt if you only use this CMD
RUN pip install --no-cache-dir -r requirements.txt

# Set the port environment variable (Cloud Run uses this)
ENV PORT 8080

# Run the web server directly using Uvicorn (Shell form for $PORT)
# --host 0.0.0.0: Makes it accessible outside the container
# --port $PORT: Uses the port defined by Cloud Run
# app:app: Specifies the location of the Flask app instance
CMD uvicorn app:app --host 0.0.0.0 --port $PORT
