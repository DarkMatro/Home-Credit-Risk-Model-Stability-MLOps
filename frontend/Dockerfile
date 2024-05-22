FROM python:3.11-slim-buster

LABEL maintainer="frolovaleh@gmail.com"

WORKDIR /app/frontend
COPY . .

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        cmake \
        build-essential \
        gcc

RUN pip install --upgrade pip && \
    pip install --ignore-installed -r requirements.txt


EXPOSE 8501

ENTRYPOINT ["streamlit","run"]
CMD ["main.py"]
