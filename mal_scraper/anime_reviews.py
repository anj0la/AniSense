"""
File: anime_reviews.py

Author: nikitperiwal
Modidied By: anj0la
Date Modified: August 2nd, 2024

This module contains methods to get anime reviews from MyAnimeList.

Source: 
    - https://github.com/nikitperiwal/MAL-Scraper
"""
import os
import re
import requests
import pandas as pd
from bs4 import BeautifulSoup


def clean_review(review: str) -> dict:
    """
    Cleans the html source containing the review,
    returning a dictionary containing the review elements.

    Args:
        review (str): The HTML dource containing review elements. 
    Returns:
        dict: The extracted review items, stored as a dictionary.
    """
    cleaned_review = list()
    for x in review.findAll(text=True):
        x = x.replace('\n', '').strip()
        if len(x) != 0 and x != 'Preliminary':
            cleaned_review.append(x)

    # inserts the cleaned review into a dict
    review_items = dict()
    review_items['Review Date'] = cleaned_review[0]
    review_items['Episodes Watched'] = cleaned_review[1]
    review_items['Username'] = cleaned_review[4]
    review_items['Review Likes'] = cleaned_review[8]
    review_items['Overall Rating'] = cleaned_review[11]
    review_items['Story Rating'] = cleaned_review[13]
    review_items['Animation Rating'] = cleaned_review[15]
    review_items['Sound Rating'] = cleaned_review[17]
    review_items['Character Rating'] = cleaned_review[19]
    review_items['Enjoyment Rating'] = cleaned_review[21]
    review_items['Review'] = "\n".join(cleaned_review[22:-5])
    return review_items


def get_all_reviews(review_link: str) -> list:
    """
    Scrapes the review webpage for all reviews, returning the cleaned reviews in a list.

    Args:
        review_link (str): The link of the page to scrape reviews from.
    Returns:
        list: A list containing all reviews data.
    """
    try:
        webpage = requests.get(review_link, timeout=10)
    except requests.exceptions.RequestException as e:
        print('Link: ', review_link)
        print('Exception: ', e, '\n')
        return None
    soup = BeautifulSoup(webpage.text, features='html.parser')
    reviews = soup.findAll('div', {'class': 'borderDark'})

    # collecting all reviews in a list after cleaning
    all_reviews = list()
    for review in reviews:
        all_reviews.append(clean_review(review))
    return all_reviews


def reviews_to_dataframe(review_list: list, anime_title: str, anime_url: str) -> pd.DataFrame:
    """
    Converts the passed review_list into a pandas DataFrame.

    Args:
        review_list (list): The list of anime reviews.
        anime_title (str): The title of the anime.
        anime_url (str): The url of the anime.
    
    Returns:
        pd.DataFrame: The DataFrame containing all reviews.
    """
    columns = ['Review Date', 'Episodes Watched', 'Username', 'Review Likes',
               'Overall Rating', 'Story Rating', 'Animation Rating', 'Sound Rating',
               'Character Rating', 'Enjoyment Rating', 'Review']
    all_data = list()
    for curr in review_list:
        data = []
        for y in columns:
            data.append(curr.get(y, ''))
        all_data.append(data)

    # creating the DataFrame
    df = pd.DataFrame(all_data, columns=columns)
    df['Anime Title'] = anime_title
    df['Anime URL'] = anime_url
    columns = ['Anime Title', 'Anime URL']+columns
    df = df[columns]
    return df


def get_anime_review(anime_title: str, anime_url: str, save_csv: bool = True, csv_dir: str = 'data/reviews/') -> pd.DataFrame:
    """
    Gets a review from the specified anime title and url, storing the review in a pandas DataFrame.
    It saves the DataFrame as a csv file if save_csv is True, and returns the DataFrame.

    Args:
        anime_title (str): The title of the anime.
        anime_url (str): The url of the anime.
        save_csv (bool): Determines whether to save the DataFrame as a CSV file. Defaults to True.
        csv_dir (str): The directory to save the CSV file in. Defaults to 'data/recommendation/'.
    
    Returns:
        pd.DataFrame: The DataFrame containing recommendations for the anime.
    """
    anime_url = anime_url + '/reviews'
    reviews = get_all_reviews(anime_url)
    df = reviews_to_dataframe(reviews, anime_title, anime_url)

    # saving the DataFrame
    if save_csv:
        if not os.path.exists(csv_dir):
            os.mkdir(csv_dir)
        anime_title = re.sub(r'[^a-zA-Z0-9]', '', anime_title)
        csv_filename = f'anime_review_{anime_title}.csv'
        full_name = os.path.join(csv_dir, csv_filename)
        df.to_csv(full_name, index=False)
    return df


def get_all_anime_reviews(anime_df: pd.DataFrame, save_csv: bool = True, save_individual: bool = False, csv_dir: str = 'data/') -> pd.DataFrame:
    """
    Gets reviews from all anime in the passed DataFrame, storing and returning the DataFrame.
    It saves the DataFrame as a csv file if save_csv is True.

    Args:
        anime_df (pd.DataFrame): The DataFrame containing anime titles and URLs.
        csv_dir: directory to store the csv file in.
        save_csv (bool): Determines whether to save the DataFrame as a CSV file. Defaults to True.
        save_individual (bool): Determines whether to save the individual anime recommendation as csv or not. Defaults to False.
        csv_dir (str): The directory to save the CSV file in. Defaults to 'data/'.
    
    Returns:
        pd.DataFrame: The DataFrame containing recommendations for all anime.
    """
    iter_data = zip(list(anime_df['Anime Title']), list(anime_df['MAL Link']))
    df = None
    for title, url in iter_data:
        if df is None:
            df = get_anime_review(title, url, save_csv=save_individual)
        else:
            df = df.append(get_anime_review(title, url, save_csv=save_individual))

    # saving the DataFrame
    if save_csv:
        if not os.path.exists(csv_dir):
            os.mkdir(csv_dir)
        csv_filename = f'anime_reviews_mal.csv'
        full_name = os.path.join(csv_dir, csv_filename)
        df.to_csv(full_name, index=False)

    return df
