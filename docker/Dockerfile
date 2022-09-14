FROM python:3.9
WORKDIR /directed_sentiment
COPY requirements.txt ./requirements.txt
RUN pip install -r requirements.txt
EXPOSE 8501
COPY . /directed_sentiment
ENTRYPOINT [ "streamlit", "run" ]
CMD ["directed_sent.app.py"]