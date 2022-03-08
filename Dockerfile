FROM python:3-slim AS builder
COPY packages.txt /tmp/ 
COPY requirements.txt /tmp/
RUN apt-get update && xargs apt-get install -y </tmp/packages.txt
RUN pip install --upgrade pip \
 && pip install --no-cache-dir --user -r /tmp/requirements.txt


FROM python:3-slim AS app
COPY --from=builder /root/.local /root/.local
WORKDIR app
COPY . .
EXPOSE 8501
ENV PATH=/root/.local/bin:$PATH
ENTRYPOINT ["streamlit", "run"]
CMD ["main.py"]
