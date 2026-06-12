# Use a lightweight official Python runtime
FROM python:3.10-slim

# Set up a working directory inside the container
WORKDIR /code

# Copy the requirements file and install dependencies
COPY ./requirements.txt /code/requirements.txt
RUN pip install --no-cache-dir --upgrade -r /code/requirements.txt

# Copy all your local scripts and model checkpoints into the container
COPY . .

# Grant write permissions to the home directory (Hugging Face security requirement)
RUN mkdir -p /.cache && chmod -R 777 /.cache

# Start the FastAPI app using Uvicorn on Hugging Face's mandatory port 7860
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "7860"]