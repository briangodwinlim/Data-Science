FROM python:3.9

WORKDIR /
COPY . .

RUN apt-get -y update
RUN apt-get update && apt-get install -y python3 python3-pip
RUN python3 -m pip install --upgrade pip
RUN pip3 install -r requirements.txt

CMD gunicorn app:server