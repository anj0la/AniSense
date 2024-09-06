"""
File: lemmatize_text.py

Author: Anjola Aina
Date Modified: May 30th, 2024

Description:
    This file contains the functions needed to lemmatize text. It is used to reduce tokens into their base form.

Functions:
    lemmatize_text(text: str) -> str: Lemmatizes each token in some text into its base form, using its POS tag.
    
Sources:
    To create the lemmatize_text function, this Geeks for Geeks source was used as a guide: https://www.geeksforgeeks.org/python-lemmatization-approaches-with-examples/
"""
from nltk.tokenize import word_tokenize
from nltk import pos_tag
from nltk.corpus import wordnet
from nltk.stem import WordNetLemmatizer

def _get_pos_tag(nltk_tag):
    """
    Gets the POS tag.

    Args:
        nltk_tag (_type_): The tag for a specific token in a sentence.

    Returns:
        (str | None): The tag type (or None if not in the tag dict).
    """
    tag_dict = {'J': wordnet.ADJ, 'V': wordnet.VERB, 'N': wordnet.NOUN, 'R': wordnet.ADV}
    return tag_dict.get(nltk_tag[0], None)

def lemmatize_text(text):
    """
    Lemmatizes each token in some text into its base form, using its POS tag, determined by a helper function to get the pos tag.

    Args:
        text (str): The text to be lemmatized.

    Returns:
        str: The lemmatized text.
    """
    text_pos_tagged = pos_tag(word_tokenize(text))
    lemmatized_text = ' '.join(WordNetLemmatizer().lemmatize(word, _get_pos_tag(tag)) if _get_pos_tag(tag) else word for word, tag in text_pos_tagged)
    return lemmatized_text