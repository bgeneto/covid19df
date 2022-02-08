FROM python:3.9

COPY requirements.txt /
COPY packages.txt /

RUN apt-get update && xargs apt-get install -y </packages.txt
RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r /requirements.txt

WORKDIR /app

COPY ./covid19df.py /app/

EXPOSE 8501

ENTRYPOINT ["streamlit", "run"]

CMD ["covid19df.py"]
