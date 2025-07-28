FROM python:3.10

WORKDIR /app

COPY . .

# Install manually
RUN pip install flask shap lime dice-ml xgboost scikit-learn matplotlib numpy pandas pdfminer.six spacy joblib

# Download spaCy model if needed
RUN python -m spacy download en_core_web_sm

EXPOSE 5000

CMD ["python", "app.py"]
