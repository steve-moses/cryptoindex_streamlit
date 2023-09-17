# Use an official Python runtime as the parent image
FROM python:3.8-slim-buster

# Set the working directory in the container
WORKDIR /app

# Copy the local files to the container
COPY ./crypto_index_app.py .
COPY ./data.csv .
COPY ./xgb_model.joblib .

# Upgrade pip
RUN pip install --upgrade pip
RUN apt-get update && apt-get install -y gcc g++ build-essential
# Install any needed packages specified in requirements.txt
RUN pip install --trusted-host pypi.python.org streamlit pandas xgboost joblib


# Install any needed packages specified in requirements.txt
RUN pip install --trusted-host pypi.python.org streamlit pandas xgboost plotly requests


# Make port 8501 available to the world outside this container
EXPOSE 8501

# Define environment variable for Streamlit
ENV STREAMLIT_SERVER_PORT=8501

# Run the app when the container launches
CMD ["streamlit", "run", "crypto_index_app.py"]
