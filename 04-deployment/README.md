## Comands ##
# pip freezee | grep scikit-learn 
# pipenv install scikit-learn==v flask
# pipenv install gunicorn
# gunicorn --bind=0.0.0.0:9696 predict:app
## Docker commands
```bash
    docker build -t ride-duration-prediction-service:v1 .
```

```bash
    docker run -it --rm -p 9696:9696 ride-duration-prediction-service:v1
```