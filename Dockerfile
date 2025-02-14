FROM python:3.8

ENV PORT 8080
ENV HOSTDIR 0.0.0.0

EXPOSE 8080

RUN apt-get update -y && \
    apt-get install -y python3-pip

COPY ./requirements.txt /app/requirements.txt

WORKDIR /app

RUN pip3 install -r requirements.txt

COPY . /app


ENTRYPOINT ["python3", "app.py"]
