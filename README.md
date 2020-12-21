## TopicClassifier

## Overview

This project is to classify the topics into the matched categories. 
In this project, the Gensim, NLTK, Spacy, Transformers, Sklearn, Tensorflow frameworks are used to estimate the title 
and the content. 
Also, a pre-trained model which is part of a model that is trained on 100 billion words from the Google News Dataset is 
used for Natural Language Processing.

## Structure

- src

    The main source code for pre processing the text, extraction of their features.
    
- utils

    * The pre-trained model for NLP
    * The source code for management of the folders and files in this project
    
- app

    The main execution file

- requirements

    All the dependencies for this project
    
- settings

    Several settings including the model path and some correlation coefficients

## Installation

- Environment

    Ubuntu 18.04, Windows 10, Python 3.6

- Dependency Installation

    Please go ahead to this project directory and run the following commands in the terminal
    ```
        pip3 install -r requirements.txt
        python3 -m nltk.downloader all
    ```

- Please create the "model" folder in the "utils" folder of this project directory and copy the model into the "model" folder
 
## Execution

- Please run the following command in the terminal

    ```
        python3 app.py
    ```