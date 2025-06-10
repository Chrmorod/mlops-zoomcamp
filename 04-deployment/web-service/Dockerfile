FROM python:3.9.7-slim

WORKDIR /app

RUN pip install -U pip
RUN pip install pipenv 
COPY requirements.txt .
RUN pip install -r requirements.txt



COPY [ "Pipfile", "Pipfile.lock", "./" ]

RUN pipenv install --system --deploy

COPY [ "predict.py", "lin_reg.bin", "./" ]

EXPOSE 9696

ENTRYPOINT [ "gunicorn", "--bind=0.0.0.0:9696", "predict:app" ]