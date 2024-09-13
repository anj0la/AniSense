""" import numpy as np
import pandas as pd
import torch 
from anisense.preprocess import test
from anisense.model import SentimentLSTM

def load_model(path):
    model = SentimentLSTM(17423) # Need to make sure that the vocab of loaded model is the SAME as train
    model.load_state_dict(torch.load(path, weights_only=True))
    model.eval()
    return model

def get_predicted_class(logits):
    print(logits)
    predicted_class = ''
    predicted_label = np.argmax(logits) # assume label is 1
    if predicted_label == 0:
        predicted_class = 'negative'
    elif predicted_label == 1:
        predicted_class = 'neutral'
    else:
        predicted_class = 'positive'
    return predicted_class

def predict(model, sentence):
    # defining the device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    
    # converting the sentence into a df for easier processing
    data = {'text': [sentence]}
    df = pd.DataFrame(data)
    
    processed_data = test(df)
    from sklearn.feature_extraction.text import CountVectorizer
    vectorizer = CountVectorizer()
    
    vectorized_text = vectorizer.fit_transform(processed_data).toarray().squeeze()
    print(vectorized_text)
    
    # getting the processed and encoded sentence
    length_encoded_sentence = [len(vectorized_text)]
    
    # reshaping tensor and getting the length
    tensor = torch.LongTensor(vectorized_text).to(device)
    
    tensor = tensor.unsqueeze(1).T  # reshape in form of batch, number of words
    
    tensor_length = torch.LongTensor(length_encoded_sentence).to(device)  
    
    print(tensor)
    
    print(tensor_length)
        
    # making the prediction and getting the corresponding class
    logits = model(tensor, tensor_length)
    prediction = get_predicted_class(logits.cpu().detach().numpy())
    return prediction   

if __name__ == '__main__':
    sentence = 'I hate watching My Hero Academia, it was so boring.'
    path_to_model = 'model/model_saved_state.pt'
    model = load_model(path_to_model)
    print(model)
    prediction = predict(model, sentence)
    print(f'Sentence: {sentence} \nPrediction: {prediction}') """