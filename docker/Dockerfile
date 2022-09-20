FROM python:3.9
LABEL maintainer "ekarakoylu@researchinnovations.org"
WORKDIR /directed_sentiment
COPY . /directed_sentiment
COPY requirements.txt ./requirements.txt
RUN pip install -r requirements.txt
EXPOSE 8501
ENTRYPOINT [ "streamlit", "run" ]
CMD ["directed_sent_app.py"]