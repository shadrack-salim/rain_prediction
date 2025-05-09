FROM continuumio/miniconda3:latest

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Create conda environment
RUN conda install -y flask=2.0.1 werkzeug=2.0.3 gunicorn=20.1.0 python-dotenv=0.19.1 \
    numpy=1.23.5 scipy=1.10.1 pandas=1.5.3 scikit-learn=1.3.0 joblib=1.1.0

# Copy the application
COPY . .

# Command to run the application
CMD gunicorn --bind 0.0.0.0:$PORT app:app
