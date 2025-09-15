FROM python:3.12-slim

WORKDIR /work

RUN pip install --no-cache-dir weaviate-client ipython

COPY py_check.py /work/py_check.py

CMD ["python","/work/py_check.py"]

