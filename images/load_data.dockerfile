# Use an Alpine Linux base image with Python 3.10
FROM python:3.10-alpine

# Set your Kaggle API key (you can pass it as a build argument)
ARG KAGGLE_USERNAME
ARG KAGGLE_KEY

# Install necessary packages
RUN apk update && \
    apk add --no-cache unzip && \
    pip install kaggle

# Set the working directory
WORKDIR /data

# Copy the Bash script into the container
COPY ../scripts/data_collection.sh .

# Make the script executable
RUN chmod +x data_collection.sh

# Define environment variables for Kaggle
ENV KAGGLE_USERNAME=${KAGGLE_USERNAME}
ENV KAGGLE_KEY=${KAGGLE_KEY}

# Run the script when the container starts
CMD ["/data/download_dogs_vs_cats.sh"]
