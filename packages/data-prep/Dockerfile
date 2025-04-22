# Use Python 3.13 as the base image
FROM python:3.13.3-slim

# Set the working directory inside the container
WORKDIR /app

# Copy only the pyproject.toml and uv.lock files first to leverage Docker caching
COPY pyproject.toml uv.lock /app/

# Install dependencies using uv
RUN pip install uv && uv install --no-cache-dir

# Copy the rest of the project files into the container
COPY . /app

# Set the default command to run main.py
CMD ["python", "main.py"]