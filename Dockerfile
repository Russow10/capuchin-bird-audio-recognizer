FROM python:3.11-slim

# Install system dependencies and SQLite
RUN apt-get update && \
    apt-get install -y sqlite3 libsqlite3-dev && \
    rm -rf /var/lib/apt/lists/*

# Set the working directory
WORKDIR /app

# Copy your code into the container
COPY . /app

# Install Python dependencies
RUN pip install --upgrade pip && pip install -r requirements.txt

# Run your Streamlit app
CMD ["streamlit", "run", "app.py"]
