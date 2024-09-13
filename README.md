# AniSense

AniSense is a project designed to analyze anime reviews. It gathers reviews from anime discussions sites such as MyAnimeList (MAL) to learn what words contribute most to positive sentiments. The model used for sentiment analysis aims to help users understand why a specific review is positive, negative or neutral. 

## Background

The model is based on an LSTM (long short-term memory) neural network, a type of RNN (recurrent neural network) that can handle long dependencies in sequences and tackles the vanishing gradient problem.

## Setup

### 1. Create a virtual environment in the directory of the project.
```
python3 -m venv .venv
```

### 2. Activate the virtual environment on your operating system.

On macOS/ Linux:

```
source .venv/bin/activate
```

On Windows:

```
.venv\Scripts\activate
```

### 3. Install the dependencies into your virtual environment.
```
pip install -r requirements.txt
```

### 4. Build the project by running the following code in your virtual environment.

```
hatch build
```
## Installing Project

Example: installing the project by its name. 
```
pip install AniSense
```

Example: installing the project in editable mode. 
```
pip install -e .
```