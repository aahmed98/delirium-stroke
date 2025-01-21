FROM python:3.7.7

ENV IN_DOCKER_CONTAINER=True

WORKDIR /
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

WORKDIR /home/p_eickhoff_stroke

CMD ["python", "experiments.py"]
