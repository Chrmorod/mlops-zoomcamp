## Comands ##
```bash
    pip freezee | grep scikit-learn
```
```bash
    pipenv install scikit-learn==v flask
```
```bash
    pipenv install gunicorn
```
```bash
    gunicorn --bind=0.0.0.0:9696 predict:app
```
## Docker commands
```bash
    docker build -t ride-duration-prediction-service:v1 .
```

```bash
    docker run -it --rm -p 9696:9696 ride-duration-prediction-service:v1
```
