"""
File: top_anime_list.py

Author: nikitperiwal
Modidied By: anj0la
Date Modified: July 31st, 2024

This module contains methods to get the top anime list from MyAnimeList.

Source: 
    - https://github.com/nikitperiwal/MAL-Scraper
"""

import os
import requests
import pandas as pd
from bs4 import BeautifulSoup
from bs4 import element
from concurrent.futures import ThreadPoolExecutor
from requests.exceptions import RequestException

def extract_anime_table_data(table: element.Tag) -> list:
    """
    Extracts and cleans data from an HTML table containing anime information.

    Parameters:
        table (element.Tag): An HTML table element extracted using BeautifulSoup.
        
    Returns:
        list: A list of lists, where each sublist contains cleaned data for a single anime.
               Each sublist includes the following details:
               - Ranking
               - Name
               - Link to the anime
               - Episodes, airing status, and member count
               - MAL score
    """
    cleaned_data = []
    for row in table.findAll('tr')[1:]:
        data = []
        cell_list = row.findAll('td')[:3]

        # extracting ranking
        data.append(cell_list[0].text.strip())

        # extracting name and link
        name_link = cell_list[1].findAll('a')[1]
        data.append(name_link.text)
        data.append(name_link['href'])

        # extracting episodes, airing status, and members
        info = cell_list[1].find('div', {'class': 'information di-ib mt4'}).text.split('\n')
        for item in info:
            item = item.strip()
            if item:
                data.append(item)

        # extracting MAL score
        data.append(cell_list[2].text.strip())

        cleaned_data.append(data)
    return cleaned_data

def get_anime_list_data(anime_list_link: str) -> list:
    """
    Extracts data from the top anime webpage, returning a cleaned version of the data.
    and returns it as a list of data after cleaning.

    Args:
        anime_list_link (str): The link of the webpage.
        
    Returns:
        list: The cleaned data.
    """
    try:
        webpage = requests.get(anime_list_link, timeout=10)
    except RequestException as e:
        print('Link: ', anime_list_link)
        print('Exception: ', e, '\n')
        return []
    soup = BeautifulSoup(webpage.text, features='html.parser')
    table = soup.find('table', {'class': 'top-ranking-table'})
    return extract_anime_table_data(table)


def get_top_data(anime_top_list_link: str, top_num: int = 200) -> list:
    """
    Scrapes the list of all top animes from the passed link 'anime_top_list_link'.
    The 'top_num' parameter specifies the number of top animes to scrape from. As 50 animes
    are displayed per page, the default value 200 will scrape 200 animes, from page 1 to page 4
    (200 / 5 = 4). 
    
    This function uses ThreadPoolExecutor to speed up the scraping process by creating multiple
    threads. Each thread handles the scraping of a single page of anime data concurrently,
    which significantly reduces the total scraping time.

    Args:
        anime_top_list_link (str): The base URL of the top-anime list.
        top_num (int): The number of animes to scrape details of (default is 200).

    Returns:
        list: A list of scraped data for all the specified top animes.
    """
    # threading to speed up the scraping process and collecting all threads in a list
    future_list = list()
    with ThreadPoolExecutor(max_workers=20) as executor:
        for i in range(start = 0, stop = top_num, step = 50):
            list_link = anime_top_list_link + str(i)
            future_list.append(executor.submit(get_anime_list_data, list_link))

    # collecting all the returned values from the threads
    all_returned_values = list()
    for val in future_list:
        all_returned_values.extend(val.result())

    return all_returned_values


def get_top_anime_data(num: int = 200, save_csv: bool = True, csv_dir: str = 'data/') -> pd.DataFrame:
    """
    Scrapes the list of all top animes from the passed link 'anime_top_list_link'.
    The 'num' parameter specifies the number of top animes to scrape from. As 50 animes
    are displayed per page, the default value 200 will scrape 200 animes, from page 1 to page 4
    (200 / 5 = 4). 
    
    If save_csv is set to True, it saves the dataframe to the specified csv directory.
    
    Finally, the function returns the dataframe containing the scraped data.
 
    Args:
        num (int): The number of anime to scrape details from. Defaults to 200.
        save_csv (bool): A boolean value determining whether to save the scraped file (as a dataframe) to csv. Defaults to true.
        csv_dir (str): The directory to save the csv file in. Defaults to data/.
        
    Returns:
        pd.DataFrame: The pandas dataframe containing the scraped data.
    """
    anime_top_list_link = 'https://myanimelist.net/topanime.php?limit='
    anime_data = get_top_data(anime_top_list_link, num)

    # creates the pandas DataFrame
    columns = ['Ranking', 'Anime Title', 'MAL Link', 'Airing Type and Episode',
               'Airing Time', 'No. of Members', 'MAL Score']
    df = pd.DataFrame(anime_data, columns=columns)

    # saves the dataframe to csv
    if save_csv:
        csv_file_name = f'top_{num}_anime_mal.csv'
        if not os.path.exists(csv_dir):
            os.mkdir(csv_dir)
        full_name = os.path.join(csv_dir, csv_file_name)
        df.to_csv(full_name, index=False)

    return df
