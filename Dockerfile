# syntax=docker/dockerfile:1

FROM python:3.8

WORKDIR /app
ADD requirements.txt /app/requirements.txt
# COPY requirements.txt requirements.txt
RUN pip3 install -r requirements.txt
ADD . /app

EXPOSE 5000

CMD [ "python", "app.py"]