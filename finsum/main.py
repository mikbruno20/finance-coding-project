import requests
from config import NEWSAPI_KEY, OPENAI_API_KEY
from database import create_table, save_summary
from summarizer import summarize_article


# Fetch news articles from NewsAPI

def fetch_news():
    url = "https://newsapi.org/v2/everything"
    params = {
        "q": "AI or robotics",
        "language": "en",
        "sortBy": "publishedAt",
        "pageSize": 10,
        "apiKey": NEWSAPI_KEY,  
        "sources": " bbc-news , bloomberg,cnn , financial-times , fortune , google-news , reuters , the-wall-street-journal ",
        "excludeDomains": "example.com,spamnews.com"
    }
    try:
        response = requests.get(url, params=params)
        response.raise_for_status()
        data = response.json()
        return data.get("articles", [])
    
    except requests.exceptions.HTTPError as http_err:
        print(f"HTTP error occurred: {http_err}")

def main():

    create_table()
    articles = fetch_news()
    for article in articles:
        title = article.get("title", "No title available")
        content = article.get("content", "No content available")
        source = article.get("source", {}).get("name", "Unknown source")
        published_at = article.get("publishedAt", "Unknown date")


        summary = summarize_article(content)
        save_summary(title, summary, source, published_at)
        print(f"Title: {title}\nSummary: {summary}\nSource: {source}\nPublished At: {published_at}\n{'-'*80}")



if __name__ == "__main__":
    main()












