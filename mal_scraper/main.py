from top_anime_list import get_top_anime_data
from anime_reviews import get_all_anime_reviews
from top_anime_details import get_all_anime_data
from anime_recommendations import get_all_anime_recs


def main():
    """
    Scrapes the top-anime list, their details, reviews and recommendations.

    """
    anime_df = get_top_anime_data(num=1000)
    get_all_anime_data(anime_df)
    get_all_anime_reviews(anime_df)
    get_all_anime_recs(anime_df)


if __name__ == "__main__":
    main()
