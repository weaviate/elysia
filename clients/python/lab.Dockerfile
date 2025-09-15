FROM python:3.12-slim

ENV PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /work

RUN pip install --no-cache-dir weaviate-client jupyterlab

COPY py_check.py /work/py_check.py

EXPOSE 8888

CMD ["jupyter","lab","--ServerApp.token=","--ServerApp.password=","--ServerApp.allow_remote_access=True","--ServerApp.ip=0.0.0.0","--ServerApp.port=8888"]

