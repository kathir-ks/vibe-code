# Use the official Python 3.11 slim image
FROM python:3.11-slim

# Set the working directory
WORKDIR /app

# Copy the application code into the container
COPY . /app

# Install required dependencies, including Quart, Hypercorn, and other needed packages
RUN pip install quart google-cloud-firestore google-generativeai hypercorn

# Set environment variable for the port
ENV PORT=8080

# Set the entrypoint to use Hypercorn (async server)
CMD ["hypercorn", "app:app", "--bind", "0.0.0.0:8080"]
