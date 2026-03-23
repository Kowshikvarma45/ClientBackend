import requests
import feedparser
from bs4 import BeautifulSoup
import random
import time
from datetime import datetime
import logging

logger = logging.getLogger(__name__)

class NewsScraper:
    def __init__(self):
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }

    def fetch_google_news_rss(self, query):
        """
        Fetches news from Google News RSS feed.
        Reliable, fast, and aggregates multiple sources.
        """
        try:
            # Encode query
            encoded_query = requests.utils.quote(query)
            rss_url = f"https://news.google.com/rss/search?q={encoded_query}+stock+news+when:7d&hl=en-US&gl=US&ceid=US:en"
            
            feed = feedparser.parse(rss_url)
            
            news_items = []
            for entry in feed.entries[:8]:
                # Attempt to find image in description if available (Google often strips it in RSS)
                # But we can try to extract source
                
                source = entry.source.title if hasattr(entry, 'source') else "Google News"
                
                item = {
                    "title": entry.title,
                    "link": entry.link,
                    "pubDate": entry.published_parsed, # struct_time
                    "source": source,
                    "summary": entry.title # RSS summary is often just the title
                }
                
                # Convert struct_time to timestamp
                if item["pubDate"]:
                    item["timestamp"] = time.mktime(item["pubDate"])
                else:
                    item["timestamp"] = time.time()
                    
                news_items.append(item)
                
            return news_items
        except Exception as e:
            logger.error(f"Google RSS fetch failed: {e}")
            return []

    def scrape_yahoo_finance(self, symbol):
        """
        Directly scrapes Yahoo Finance for news and thumbnails.
        Good for thumbnails.
        """
        try:
            url = f"https://finance.yahoo.com/quote/{symbol}/news"
            response = requests.get(url, headers=self.headers, timeout=10)
            
            if response.status_code != 200:
                return []
                
            soup = BeautifulSoup(response.text, 'html.parser')
            news_items = []
            
            # Find news stream - Selectors change often, so we try a few common patterns
            # Yahoo often uses `li` inside a stream list
            articles = soup.find_all('li', class_='js-stream-content')
            
            for article in articles[:8]:
                try:
                    # Find Title
                    title_tag = article.find('h3')
                    if not title_tag: continue
                    title = title_tag.get_text()
                    
                    # Find Link
                    link_tag = article.find('a')
                    link = link_tag['href'] if link_tag else ""
                    if link and not link.startswith('http'):
                        link = f"https://finance.yahoo.com{link}"
                        
                    # Find Source
                    source_tag = article.find('div', class_='C(#959595)') or article.find('span')
                    source = source_tag.get_text() if source_tag else "Yahoo Finance"
                    
                    # Find Paragraph/Summary
                    p_tag = article.find('p')
                    summary = p_tag.get_text() if p_tag else title
                    
                    # Find Image (Tricky on Yahoo, often background-image or img tag)
                    img_tag = article.find('img')
                    thumbnail = img_tag['src'] if img_tag else None
                    
                    news_items.append({
                        "title": title,
                        "link": link,
                        "source": source,
                        "summary": summary,
                        "thumbnail": thumbnail,
                        "timestamp": time.time() # Hard to parse relative time "2 hours ago" reliably without lib
                    })
                except Exception:
                    continue
                    
            return news_items
            
        except Exception as e:
            logger.error(f"Yahoo Search failed: {e}")
            return []

    def get_news(self, symbol):
        """
        Hybrid strategy: 
        1. Try Google News RSS (Best for reliability & variety)
        2. Enhance/Fallback with generic images if needed
        """
        logger.info(f"Scraping news for {symbol}...")
        
        rss_items = self.fetch_google_news_rss(symbol)
        final_items = []
        
        # Specific Image Maps based on Stock Symbol
        # This ensures high relevance for major stocks
        stock_images = {
            "AAPL": "https://images.unsplash.com/photo-1519389950473-47ba0277781c?q=80&w=800",
            "MSFT": "https://images.unsplash.com/photo-1642132652875-884d43496f3a?q=80&w=800",
            "GOOGL": "https://images.unsplash.com/photo-1573804633927-bfcbcd909acd?q=80&w=800",
            "AMZN": "https://images.unsplash.com/photo-1523474253046-8cd2748b5fd2?q=80&w=800",
            "TSLA": "https://images.unsplash.com/photo-1617788138017-80ad40651399?q=80&w=800",
            "NVDA": "https://images.unsplash.com/photo-1629654297299-c8506221ca97?q=80&w=800",
            "META": "https://images.unsplash.com/photo-1611162617474-5b21e879e113?q=80&w=800",
            "NFLX": "https://images.unsplash.com/photo-1574375927938-d5a98e8ffe85?q=80&w=800"
        }
        
        for item in rss_items:
            # 1. Determine Image
            thumb = None
            
            # Check specific symbol map first
            if symbol in stock_images:
                thumb = stock_images[symbol]
            else:
                # Heuristic keyword matching
                keywords = item['title'].lower() + " " + item['summary'].lower()
                
                if "tech" in keywords or "ai " in keywords or "chip" in keywords:
                     thumb = "https://images.unsplash.com/photo-1518770660439-4636190af475?q=80&w=800"
                elif "oil" in keywords or "energy" in keywords:
                     thumb = "https://images.unsplash.com/photo-1500319300304-71280b2a4c14?q=80&w=800" # Better oil img
                elif "bank" in keywords or "finance" in keywords or "rates" in keywords:
                     thumb = "https://images.unsplash.com/photo-1611974765270-ca12586343bb?q=80&w=800"
                elif "pharma" in keywords or "health" in keywords:
                     thumb = "https://images.unsplash.com/photo-1532187863486-abf9dbad1b69?q=80&w=800"
                elif "auto" in keywords or "car" in keywords:
                     thumb = "https://images.unsplash.com/photo-1617788138017-80ad40651399?q=80&w=800"
                else:
                     # General Stock Market fallback
                     thumb = "https://images.unsplash.com/photo-1611974765270-ca12586343bb?q=80&w=800"
            
            final_items.append({
                "title": item['title'],
                "summary": item['summary'],
                "source": item['source'],
                "url": item['link'],
                "thumbnail": thumb,
                "time_published": item['timestamp']
            })
            
        return final_items

    def extract_article_content(self, url):
        """
        Extracts the main content of an article for the internal reader.
        """
        try:
            # Follow redirects first (Google News RSS links are redirected)
            try:
                response = requests.get(url, headers=self.headers, timeout=10, allow_redirects=True)
                final_url = response.url
                html = response.text
            except:
                return {"error": "Failed to load article"}

            soup = BeautifulSoup(html, 'html.parser')
            
            # Simple heuristic extraction (for robust extraction we'd use 'newspaper3k' library)
            # Remove scripts and styles
            for script in soup(["script", "style", "nav", "footer", "header", "aside"]):
                script.decompose()

            # Find paragraphs
            paragraphs = soup.find_all('p')
            content_blocks = []
            
            for p in paragraphs:
                text = p.get_text().strip()
                # Filter out short snippets like "Copyright", "Read more", etc.
                if len(text) > 60: 
                    content_blocks.append(text)
            
            # Join top 10 meaningful blocks
            full_text = "\n\n".join(content_blocks[:15])
            
            # Try to get high-res image from og:image
            og_image = soup.find("meta", property="og:image")
            image = og_image["content"] if og_image else None
            
            return {
                "content": full_text,
                "author": "Scraped Source", # Hard to detect reliably without dedicated lib
                "final_url": final_url,
                "image": image # Use this if it exists, else fallback to what we had
            }
            
        except Exception as e:
            logger.error(f"Extraction error: {e}")
            return {"error": str(e)}

# Test locally
if __name__ == "__main__":
    scraper = NewsScraper()
    items = scraper.get_news("AAPL")
    print(f"Found {len(items)} items")
    for item in items[:3]:
        print(item)
