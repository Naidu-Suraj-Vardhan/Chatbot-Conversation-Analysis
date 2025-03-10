# Use the official Python image from the Docker Hub
FROM python:3.9-slim

# Set the working directory in the container
WORKDIR /app

# Copy the requirements file into the container
COPY requirements.txt .

# Install the required packages
RUN pip install --no-cache-dir -r requirements.txt

# Copy the entire application code into the container
COPY . .

# Set environment variables
ENV OPENAI_API_KEY=your_api_key_here

# Expose the port that Gradio will run on
EXPOSE 7860

# Command to run the application
CMD ["python3", "app.py"]