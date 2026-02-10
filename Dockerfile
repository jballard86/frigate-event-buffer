# Use a lightweight Python image
FROM python:3.9-slim

# Set the working directory inside the container
WORKDIR /app

# Copy the dependency list and install them
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the actual application code
COPY app.py .

# Create the storage directory (Unraid will map this to your array)
RUN mkdir -p /data

# Tell Docker we use port 5050
EXPOSE 5050

# Run the app
CMD ["python", "app.py"]