FROM dmbtburke/gpt2-tedtalk:4
WORKDIR /app
RUN pip install flask transformers torch
COPY . .
CMD ["python3", "app.py"]
