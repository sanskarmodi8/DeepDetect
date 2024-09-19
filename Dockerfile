FROM python:3.8-slim-buster

WORKDIR /app

COPY . /app

RUN apt update -y && apt-get install -y && pip install -r requirements.txt

CMD ["python3", "app.py"]