# Use Python base image
FROM python:3.10

# Set working directory inside the container
WORKDIR /app

# Copy only the backend folder content
COPY cvision/backend/ /app/

# Install Python packages
RUN pip install --upgrade pip
RUN pip install flask shap lime dice-ml xgboost scikit-learn matplotlib numpy pandas pdfminer.six spacy joblib

# Optional: Download spaCy model
RUN python -m spacy download en_core_web_sm

# Expose Flask port
EXPOSE 5000

# Run Flask app
CMD ["python", "app.py"]
