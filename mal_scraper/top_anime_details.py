"""
File: top_anime_details.py

Author: nikitperiwal
Modidied By: anj0la
Date Modified: July 31st, 2024

This module contains methods to get the details of top animes from MyAnimeList.

Source: 
    - https://github.com/nikitperiwal/MAL-Scraper
"""
import os
import time
import requests
import pandas as pd
from bs4 import BeautifulSoup
from requests.exceptions import RequestException

def clean_side_panel(side_panel: list) -> dict:
    """
    Cleans the SidePanel data and stores it in a dictionary.

    Args:
        side_panel (list): The HTML elements of the side panel extracted using BeautifulSoup.
        
    Returns:
        dict: A dictionary of extracted data.
    """
    data = dict()
    for x in side_panel:
        x = x.text.strip().replace('\n', '')
        index = x.find(':')
        if index == -1:
            continue
        key, value = x[:index], x[index+1:].strip()
        data[key] = value
    return data

def get_anime_detail(anime_link: str) -> dict:
    """
    Fetches and cleans the details of a single anime from its webpage.

    Args:
        anime_link (str): The URL of the anime's webpage.
        
    Returns:
        dict: A dictionary of extracted data.
    """
    try:
        webpage = requests.get(anime_link, timeout=10)
    except RequestException as e:
        print('Link: ', anime_link)
        print('Exception: ', e, '\n')
        return None
    soup = BeautifulSoup(webpage.text, features='html.parser')
    side_panel = soup.find('td', {'class': 'borderClass'}).find('div').findAll('div')[6:]
    data = clean_side_panel(side_panel)
    data['Summary'] = soup.find('p', {'class': ''}).text
    return data

def clean_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Cleans the DataFrame passed, to make the values more readable.

    Args:
        df (pd.DataFrame): The DataFrame to clean.
        
    Returns:
        pd.DataFrame: The cleaned DataFrame.
    """
    
    # function to remove extra spaces from the cell in between items
    def clean_divide(cell: str) -> str:
        cell = cell.split(',')
        for i in range(len(cell)):
            cell[i] = cell[i].strip()
        if cell[0] == 'None found':
            return ''
        return ", ".join(cell)

    # function to remove double entries from the cell
    def remove_double(cell: str) -> str:
        cell = cell.split(', ')
        for i in range(len(cell)):
            cell[i] = cell[i][:len(cell[i]) // 2]
        return ', '.join(cell)

    # cleaning up the dataframe
    df['Genres'] = df['Genres'].apply(clean_divide)
    df['Genres'] = df['Genres'].apply(remove_double)
    df['Studios'] = df['Studios'].apply(clean_divide)
    df['Producers'] = df['Producers'].apply(clean_divide)
    df['Licensors'] = df['Licensors'].apply(clean_divide)
    df['Score'] = df['Score'].apply(lambda x: x[:4])
    df['Ranked'] = df['Ranked'].apply(lambda x: x[1:-99])
    df['Members'] = df['Members'].str.replace(',', '')
    df['Favorites'] = df['Favorites'].str.replace(',', '')
    df['Popularity'] = df['Popularity'].str.replace('#', '')

    return df


def dict_to_pandas(data_dict_list: list) -> pd.DataFrame:
    """
    Converts the passed list of dictionaries into a pandas DataFrame
    and returns the cleaned DataFrame.

    Args:
        data_dict_list (list): A list of dictionaries containing scraped data.
        
    Returns:
        pd.DataFrame: The DataFrame containing the scraped data as a table.
    """
    columns = ['Anime Title', 'MAL Url', 'English', 'Japanese', 'Type', 'Episodes',
               'Status', 'Aired', 'Premiered', 'Broadcast', 'Producers', 'Licensors',
               'Studios', 'Source', 'Genres', 'Duration', 'Rating', 'Score', 'Ranked',
               'Popularity', 'Members', 'Favorites', 'Summary']
    all_data = list()

    for curr in data_dict_list:
        data = []
        for y in columns:
            data.append(curr.get(y, ''))
        all_data.append(data)
    df = pd.DataFrame(all_data, columns=columns)
    return clean_dataframe(df)


def get_all_anime_data(anime_df: pd.DataFrame, save_csv: bool = True, csv_dir: str = 'data/', sleep_time: int = 1) -> pd.DataFrame:
    """
    Scrapes details of all anime in the DataFrame passed and returns the scraped details as a pandas DataFrame.
    
    If save_csv is set to True, it saves the dataframe to the specified csv directory.

    Args:
        anime_df (pd.DataFrame): The DataFrame containing anime titles and URLs.
        save_csv (bool): Determines whether to save the DataFrame as a CSV file. Defaults to True.
        csv_dir (str): The directory to save the CSV file in. Defaults to 'data/'.
        sleep_time (int): The time to sleep between each page request in seconds. Defaults to 1.
        
    Returns:
        pd.DataFrame: The DataFrame containing the scraped anime data.
    """
    all_data = list()
    iterative_data = zip(list(anime_df['MAL Link']), list(anime_df['Anime Title']))
    for url, name in iterative_data:
        # delays the program so that the website does not stop responding
        time.sleep(sleep_time)
        res = get_anime_detail(url)
        if res is None:
            return all_data
        else:
            res['Anime Title'] = name
            res['MAL Url'] = url
            all_data.append(res)

    df = dict_to_pandas(all_data)

    # Saves the csv.
    if save_csv:
        csv_filename = f'{len(anime_df)}_anime_details_mal.csv'
        if not os.path.exists(csv_dir):
            os.mkdir(csv_dir)
        full_name = os.path.join(csv_dir, csv_filename)
        df.to_csv(full_name, index=False)

    return df
