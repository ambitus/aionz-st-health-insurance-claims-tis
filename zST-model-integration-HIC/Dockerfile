FROM icr.io/ibmz/python:3.9
RUN mkdir /app
WORKDIR /app
COPY requirements.txt .
RUN pip3 install -r requirements.txt
COPY Results/ClaimsAmountResults.csv ./Results/
RUN mkdir /ui
COPY ui ./ui/
#handle 404 from browser router
COPY ui/index.html ./ui/404.html

COPY constants.py .
COPY main.py .

# Show python console output
ENV PYTHONUNBUFFERED=1
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "80", "--workers", "2"]
