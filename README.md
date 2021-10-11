# GPT2 TedTalk

[![Run on Ainize](https://ainize.ai/images/run_on_ainize_button.svg)](https://ainize.ai/dmtburke/ainize-gpt2-tedtalk?branch=master)


This project generates Ted Talks using GPT-2 model.

Fine tuning data: [Ted Talk Transcripts](https://www.kaggle.com/rounakbanik/ted-talks)

### How I made it
See my [colab notebook](https://colab.research.google.com/drive/1TyWJjnca19TegycuqSiSc0bC51RLyo3t?usp=sharing)! This is the server side of this application, view the front end (and demo!) [here](https://ainize.ai/dmtburke/ainize-ted-talk-demo?branch=main).



### How to use

    git clone https://github.com/dmtburke/ainize-gpt2-tedtalk

    cd ainize-gpt2-tedtalk

    docker build --tag {project-name}:{tag} . 

    docker run -p 5000:5000 {project-name}:{tag} 

### Post parameter

    base_text: The base of your Ted Talk.
    length: The length of the text


### Output format

    {"prediction": generated text}


### Input example


    curl -X POST "https://master-ainize-gpt2-tedtalk-dmtburke.endpoint.ainize.ai/predict" -H "accept: application/json" -H "Content-Type: multipart/form-data" -F "base_text=When I was studying machine learning" -F "length=100"

## * With swagger *

API page: [Ainize](https://ainize.ai/dmtburke/ainize-gpt2-tedtalk?branch=master)
