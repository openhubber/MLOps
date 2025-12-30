FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
RUN apt-get update && apt-get install -y git

COPY . .

ENV DATA_PATH=./data/creditcard.csv
ENV MODEL_SAVE_PATH=./model/saved_models/model.pkl
ENV MLFLOW_TRACKING_URI=http://172.17.0.6:5000

# CMD ["python", "model/model.py"]
CMD ["python", "api/app.py"]
