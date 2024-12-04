FROM nvidia/cuda:11.8.0-base-ubuntu22.04
WORKDIR /root/src/

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.10 python3-pip python3-dev build-essential cmake zlib1g-dev curl \
    gcc libc-dev libffi-dev g++ swig bash netcat-traditional \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

RUN python3 -m pip install --upgrade pip setuptools

COPY . src
RUN pip install poetry

WORKDIR /root/src/src
RUN POETRY_VIRTUALENVS_CREATE=false poetry install && rm -rf /root/.cache/pypoetry/*

EXPOSE 9090

RUN chmod +x /root/src/src/bin/start-app.sh
ENTRYPOINT ["/root/src/src/bin/start-app.sh"]
