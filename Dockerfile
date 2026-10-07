# Start from a lightweight official Python image
FROM python:3.10-slim

# Set the working directory inside the container
WORKDIR /app

# Copy requirement list and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy your source code into the container
COPY . .

# Specify the command to run your script
CMD ["python", "main.py"]