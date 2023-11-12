#This Dockerfile is done exclusively to check if main.py works with requirements.txt

FROM tensorflow/tensorflow:2.8.0

WORKDIR /app

COPY requirements.txt .
RUN apt-get update &&  \
    apt-get install -y gcc && \
    pip install --upgrade pip && \
    python -m pip install -r requirements.txt && \
    pip install pmdarima==1.8.5

COPY csv_popularity ./csv_popularity
COPY main.py .

# VOLUME /plots/results

# RUN ls csv_popularity

RUN python main.py