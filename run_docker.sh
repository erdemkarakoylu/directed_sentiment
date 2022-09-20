sudo docker image build -t direct_sentiment:latest -f docker/Dockerfile .
sudo docker run -dp 8501:8501 direct_sentiment
