# AniSense

AniSense is a project designed to analyze anime reviews. It gathers reviews from anime discussions sites such as MyAnimeList (MAL) to learn what words contribute most to positive sentiments. The model used for sentiment analysis aims to help users understand why a specific review is positive, negative or neutral. 

## Background

The model is based on an LSTM (long short-term memory) neural network, a type of RNN (recurrent neural network) that can handle long dependencies in sequences and tackles the vanishing gradient problem, making it a neural network to be used for classification problems. 

## Setup

### 1. Create a virtual environment in the directory of the project.
```
python3 -m venv venv
```

### 2. Activate the virtual environment on your operating system.

On macOS/ Linux:

```
source venv/bin/activate
```

On Windows (cmd):

```
.venv\Scripts\activate.bat
```

On Windows (Powershell):

```
.venv\Scripts\Activate.ps1
```

### 3. Install the dependencies into your virtual environment.
```
pip install -r requirements.txt
```

### 4. Build the project by running the following code in your virtual environment.

```
python setup.py sdist bdist_wheel
```

### 5. Install the project. You can do so by using the name 'AniSense', or installing it in editable mode with -e to make changes.

Example: installing the project by its name. 
```
pip install AniSense
```

Example: installing the project in editable mode. 
```
pip install -e .
```

## Possible Expansions

1. Adding more reviews from other websites to provide more training data to the model. This requires making custom web scrapers for the other websites.
2. Making a recommendation system based on the reviews and the recommendations. The idea is to start with a simple KNN classifier, and then use a more complicated and stronger machine learning algorithm like MF and deep neural networks.