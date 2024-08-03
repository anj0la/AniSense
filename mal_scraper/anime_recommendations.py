"""
File: anime_recommendations.py

Author: nikitperiwal
Modidied By: anj0la
Date Modified: August 2nd, 2024

This module contains methods to get the details of anime recommendations from MyAnimeList.

Source: 
    - https://github.com/nikitperiwal/MAL-Scraper
"""
import os
import re
import requests
import pandas as pd
from bs4 import BeautifulSoup
from requests.exceptions import RequestException

def clean_recs(recs: str) -> list:
    """
    Cleans the HTML source, returning a list containing the recommendations.

    Args:
        recs (str): The recommendations in the HTML format.
    
    Returns:
        list: The list containing the extracted recommendations. 
    """
    cleaned_recs = list()
    for rec in recs:
        name = rec.find('div', {'style': 'margin-bottom: 2px;'}).text.replace('\n', '').strip()[:-13]
        try:
            num = int(rec.find('div', {'class': 'spaceit'}).text[24:-10]) + 1
        except AttributeError:
            num = 1
        cleaned_recs.append([name, num])
    return cleaned_recs


def get_recs(recs_url: str) -> list:
    """
    Scrapes the user recommendation webpage for all recommendations,
    returning them in a list.

    Args:
        recs_url (str): The link of the page to scrape the recommendations from.
    
    Returns:
        list: A list containing information about all recommendations.
    """
    try:
        webpage = requests.get(recs_url, timeout=10)
    except RequestException as e:
        print('Link: ', recs_url)
        print('Exception: ', e, '\n')
        return None
    soup = BeautifulSoup(webpage.text, features='html.parser')

    # getting the recommendation part from the webpage
    recs = []
    for x in soup.findAll('div', {'class': 'borderClass'}):
        if len(x.findAll('div', {'class': 'borderClass'})) > 0:
            recs.append(x)
    recs = clean_recs(recs)
    return recs


def recs_to_dataframe(recs_list: list, anime_title: str, anime_url: str) -> pd.DataFrame:
    """
    Converts the passed recs_list into a pandas DataFrame.

    Args:
        recs_list (list): The list of anime recommendations.
        anime_title (str): The title of the anime.
        anime_url (str): The url of the anime.
    
    Returns:
        pd.DataFrame: The DataFrame containing all recommendations.
    """
    columns = ['Anime Title', 'Anime URL', 'Recommended Title', 'No. of Recommendations']
    dataframe = pd.DataFrame(recs_list, columns=columns[2:])
    dataframe['Anime Title'] = anime_title
    dataframe['Anime URL'] = anime_url
    dataframe = dataframe[columns]
    return dataframe


def get_anime_recs(anime_title: str, anime_url: str, save_csv: bool = True, csv_dir: str ='data/recommendation/') -> pd.DataFrame:
    """
    Gets recommendations from the specified anime title and url, storing the recommendation in a pandas DataFrame.
    It saves the DataFrame as a csv file if save_csv is True, and returns the DataFrame.

    Args:
        anime_title (str): The title of the anime.
        anime_url (str): The url of the anime.
        save_csv (bool): Determines whether to save the DataFrame as a CSV file. Defaults to True.
        csv_dir (str): The directory to save the CSV file in. Defaults to 'data/recommendation/'.
    
    Returns:
        pd.DataFrame: The DataFrame containing recommendations for the anime.
    """
    anime_url = anime_url + '/userrecs'
    recs = get_recs(anime_url)
    df = recs_to_dataframe(recs, anime_title, anime_url)

    # saving the DataFrame
    if save_csv:
        if not os.path.exists(csv_dir):
            os.mkdir(csv_dir)
        anime_title = re.sub(r'[^a-zA-Z0-9]', '', anime_title)
        csv_filename = f'anime_recommendations_{anime_title}.csv'
        full_name = os.path.join(csv_dir, csv_filename)
        df.to_csv(full_name, index=False)

    return df


def get_all_anime_recs(anime_df: pd.DataFrame, save_csv: bool = True, save_individual: bool = False, csv_dir: str = 'data/') -> pd.DataFrame:
    """
    Gets recommendations for all anime in the DataFrame passed, storing all recommendations in a pandas DataFrame.
    It saves the DataFrame as a csv file if save_csv is True

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
            df = get_anime_recs(title, url, save_csv=save_individual)
        else:
            df = df.append(get_anime_recs(title, url, save_csv=save_individual))

    # saving the DataFrame
    if save_csv:
        if not os.path.exists(csv_dir):
            os.mkdir(csv_dir)
        csv_filename = f'MAL Anime Recommendations.csv'
        full_name = os.path.join(csv_dir, csv_filename)
        df.to_csv(full_name, index=False)

    return df
