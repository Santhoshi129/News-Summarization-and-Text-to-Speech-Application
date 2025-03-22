import requests
from bs4 import BeautifulSoup
import re
from typing import List, Dict, Any, Optional
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import time
import logging
import random
from newspaper import Article, Config

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Download NLTK resources
try:
    nltk.download(['punkt', 'stopwords', 'wordnet', 'vader_lexicon'], quiet=True)
except Exception as e:
    logger.warning(f"NLTK download issue: {str(e)}")

class NewsExtractor:
    """Efficient news extraction focused on relevant company news"""
    
    def __init__(self):
        self.user_agents = [
            'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
            'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/92.0.4515.107 Safari/537.36',
            'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:90.0) Gecko/20100101 Firefox/90.0',
            'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/15.0 Safari/605.1.15',
            'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/96.0.4664.110 Safari/537.36 Edg/96.0.1054.62'
        ]
        
        self.config = Config()
        self.config.browser_user_agent = self._get_random_ua()
        self.config.request_timeout = 30  # Increased timeout
        
        # Expanded search URLs for better coverage
        self.search_urls = [
            ("https://news.google.com/search?q=company%20{}&hl=en-US&gl=US&ceid=US:en", "html"),
            ("https://finance.yahoo.com/quote/{}?p={}", "html"),
            ("https://www.marketwatch.com/search?q={}&m=Keyword&rpp=100", "html"),
            ("https://www.reuters.com/search/news?blob={}", "html"),
            ("https://www.investing.com/search/?q={}", "html")
        ]

    def _get_random_ua(self):
        """Get a random user agent string"""
        return random.choice(self.user_agents)
    
    def get_search_results(self, company: str, num_articles: int = 10) -> List[str]:
        """Get search results optimized for company relevance"""
        all_urls = []
        company_ticker = self._guess_ticker(company)
        
        # Enhanced search terms for better results
        search_terms = [
            f"{company} financial news",
            f"{company} company news",
            f"{company} latest news",
            f"{company} business",
            f"{company} stock news",
            f"{company} market update",
            f"{company} press release",
            f"{company} earnings",
            f"{company} industry news"
        ]
        
        for search_term in search_terms:
            encoded_term = requests.utils.quote(search_term)
            for url_template, parser_type in self.search_urls:
                try:
                    formatted_url = url_template.format(encoded_term, company_ticker)
                    
                    headers = {
                        'User-Agent': self._get_random_ua(),
                        'Accept-Language': 'en-US,en;q=0.5',
                        'Referer': 'https://www.google.com/',
                        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
                        'Connection': 'keep-alive',
                        'DNT': '1',
                        'Upgrade-Insecure-Requests': '1'
                    }
                    
                    # Add delay to avoid rate limits
                    time.sleep(random.uniform(0.5, 1.5))
                    
                    response = requests.get(
                        formatted_url,
                        headers=headers,
                        timeout=20
                    )
                    
                    if response.status_code == 200:
                        urls = self._parse_html_search_results(response.text, url_template)
                        if urls:
                            all_urls.extend(urls)
                            logger.info(f"Found {len(urls)} URLs from {url_template}")
                            if len(all_urls) >= num_articles * 4:  # Get 4x more URLs than needed for filtering
                                break
                
                except Exception as e:
                    logger.warning(f"Search attempt failed for {url_template}: {str(e)}")
        
        # Filter for more relevant results by checking if company name appears in URL or domain is reputable
        company_keywords = company.lower().split()
        filtered_urls = []
        
        # List of reputable financial news domains
        reputable_domains = [
            "reuters.com", "bloomberg.com", "ft.com", "wsj.com", "cnbc.com", 
            "marketwatch.com", "investing.com", "finance.yahoo.com", "fool.com",
            "businessinsider.com", "forbes.com", "seekingalpha.com", "barrons.com",
            "economist.com", "morningstar.com", "benzinga.com", "zacks.com",
            "thestreet.com", "investopedia.com", "nasdaq.com", "bbc.com",
            "apnews.com", "nytimes.com"
        ]
        
        # Prioritize URLs with company name or from reputable sources
        priority_urls = []
        secondary_urls = []
        
        for url in all_urls:
            url_lower = url.lower()
            
            # Check if URL contains company name or keywords
            has_company_reference = any(keyword in url_lower for keyword in company_keywords)
            
            # Check if URL is from reputable financial news source
            is_reputable = any(domain in url_lower for domain in reputable_domains)
            
            if has_company_reference and is_reputable:
                priority_urls.append(url)
            elif has_company_reference or is_reputable:
                secondary_urls.append(url)
            elif '/news/' in url_lower or '/article/' in url_lower:
                filtered_urls.append(url)
        
        # Combine lists with priority order
        combined_urls = priority_urls + secondary_urls + filtered_urls
        
        # Remove duplicates
        unique_urls = []
        seen = set()
        for url in combined_urls:
            if url not in seen:
                seen.add(url)
                unique_urls.append(url)
        
        logger.info(f"After filtering: {len(unique_urls)} unique URLs (priority: {len(priority_urls)}, secondary: {len(secondary_urls)}, other: {len(filtered_urls)})")
        
        # Return more URLs than needed to ensure enough articles after content extraction
        return unique_urls[:num_articles * 3]

    def _guess_ticker(self, company: str) -> str:
        """Make a simple guess at company ticker symbol"""
        common_tickers = {
            "apple": "AAPL",
            "microsoft": "MSFT",
            "amazon": "AMZN",
            "google": "GOOGL",
            "alphabet": "GOOGL",
            "tesla": "TSLA",
            "facebook": "META",
            "meta": "META",
            "netflix": "NFLX",
            "nvidia": "NVDA",
            "ibm": "IBM",
            "intel": "INTC",
            "oracle": "ORCL",
            "cisco": "CSCO",
            "adobe": "ADBE",
            "salesforce": "CRM",
            "paypal": "PYPL",
            "amd": "AMD",
            "qualcomm": "QCOM",
            "disney": "DIS",
            "coca-cola": "KO",
            "pepsi": "PEP",
            "walmart": "WMT",
            "target": "TGT",
            "nike": "NKE",
            "mcdonalds": "MCD",
            "johnson & johnson": "JNJ",
            "pfizer": "PFE",
            "moderna": "MRNA",
            "exxon": "XOM",
            "chevron": "CVX",
            "ford": "F",
            "general motors": "GM",
            "general electric": "GE",
            "boeing": "BA",
            "lockheed martin": "LMT",
            "jp morgan": "JPM",
            "bank of america": "BAC",
            "goldman sachs": "GS",
            "visa": "V",
            "mastercard": "MA"
        }
        
        company_lower = company.lower()
        
        for key, ticker in common_tickers.items():
            if key in company_lower:
                return ticker
                
        # Simple heuristic: use first letters of company name
        words = company.split()
        if len(words) > 1:
            return ''.join(word[0].upper() for word in words if word)
        else:
            return company[:4].upper()

    def _parse_html_search_results(self, html: str, source: str) -> List[str]:
        """Parse HTML search results with improved relevance filtering"""
        try:
            soup = BeautifulSoup(html, 'html.parser')
            links = []
            
            # Google News parsing
            if 'news.google.com' in source:
                article_links = soup.find_all('a', {'class': 'VDXfz'})
                if article_links:
                    for link in article_links:
                        href = link.get('href', '')
                        if href.startswith('./articles/'):
                            # Convert relative link to absolute
                            full_url = f"https://news.google.com{href[1:]}"
                            links.append(full_url)
                
                # Alternate selector if first method fails
                if not links:
                    for a in soup.find_all('a', href=True):
                        href = a['href']
                        if './articles/' in href:
                            full_url = f"https://news.google.com{href[1:]}"
                            links.append(full_url)
            
            # Yahoo Finance parsing
            elif 'yahoo.com' in source:
                for a in soup.find_all('a', href=True):
                    href = a['href']
                    if '/news/' in href and href.startswith('http'):
                        links.append(href)
                    elif '/news/' in href and href.startswith('/'):
                        full_url = f"https://finance.yahoo.com{href}"
                        links.append(full_url)
            
            # MarketWatch parsing
            elif 'marketwatch.com' in source:
                for a in soup.find_all('a', href=True):
                    href = a['href']
                    if '/story/' in href and href.startswith('http'):
                        links.append(href)
                    elif '/story/' in href and href.startswith('/'):
                        full_url = f"https://www.marketwatch.com{href}"
                        links.append(full_url)
            
            # Reuters parsing
            elif 'reuters.com' in source:
                for a in soup.find_all('a', href=True):
                    href = a['href']
                    if '/article/' in href and href.startswith('http'):
                        links.append(href)
                    elif '/article/' in href and href.startswith('/'):
                        full_url = f"https://www.reuters.com{href}"
                        links.append(full_url)
            
            # Investing.com parsing
            elif 'investing.com' in source:
                for a in soup.find_all('a', href=True):
                    href = a['href']
                    if '/news/' in href and href.startswith('http'):
                        links.append(href)
                    elif '/news/' in href and href.startswith('/'):
                        full_url = f"https://www.investing.com{href}"
                        links.append(full_url)
            
            # Generic news article detection
            if not links:
                for a in soup.find_all('a', href=True):
                    href = a['href']
                    if href.startswith('http') and ('/news/' in href or '/article/' in href or '/story/' in href):
                        links.append(href)
            
            # Log how many links we found
            logger.info(f"Found {len(links)} links from source: {source}")
            return links[:30]  # Return more links than needed
            
        except Exception as e:
            logger.warning(f"HTML parsing error: {str(e)}")
            return []

    def extract_article_content(self, url: str) -> Optional[Dict[str, Any]]:
        """Extract article content with improved relevance checking"""
        try:
            # Add random delay to avoid rate limiting
            time.sleep(random.uniform(0.5, 1.0))
            
            # Set random user agent for each request
            self.config.browser_user_agent = self._get_random_ua()
            
            # Use newspaper3k for extraction
            article = Article(url, config=self.config)
            article.download()
            article.parse()
            
            # Skip very short articles
            if len(article.text) < 100:  # Reduced minimum length
                logger.info(f"Skipping short article: {url} ({len(article.text)} chars)")
                return None
                
            # Try to get publication date
            try:
                article.publish_date_string = article.publish_date.strftime('%Y-%m-%d') if article.publish_date else "Unknown"
            except:
                article.publish_date_string = "Unknown"
                
            return {
                "title": article.title,
                "content": article.text,
                "summary": article.meta_description or article.text[:150] + "...",
                "url": url,
                "source": url.split('/')[2],
                "publish_date": article.publish_date_string
            }
                
        except Exception as e:
            logger.warning(f"Article extraction failed for {url}: {str(e)}")
            return None

    def get_company_news(self, company: str, num_articles: int = 10) -> List[Dict[str, Any]]:
        """Get relevant company news articles with content validation"""
        # Get more search results than requested to ensure we have enough after filtering
        urls = self.get_search_results(company, num_articles * 3)
        articles = []
        
        logger.info(f"Fetching content for {len(urls)} URLs for company: {company}")
        
        # Extract content from URLs
        successful_extractions = 0
        for url in urls:
            try:
                article = self.extract_article_content(url)
                if article:
                    # Check if article content is relevant to the company
                    if self._is_article_relevant(article, company):
                        articles.append(article)
                        successful_extractions += 1
                        logger.info(f"Added relevant article {successful_extractions}: {article.get('title', 'No title')}")
                    else:
                        logger.info(f"Skipped irrelevant article: {article.get('title', 'No title')}")
                
                # Break early if we've reached our target
                if len(articles) >= num_articles:
                    break
                    
            except Exception as e:
                logger.warning(f"Article extraction error for {url}: {str(e)}")
                continue
        
        # If we found fewer articles than requested, try backup sources
        if len(articles) < num_articles:
            logger.info(f"Found only {len(articles)} articles, trying backup sources")
            try:
                # Try with more generic business search
                backup_urls = self._get_backup_urls(company)
                logger.info(f"Using {len(backup_urls)} backup URLs for company: {company}")
                
                for url in backup_urls:
                    try:
                        article = self.extract_article_content(url)
                        if article and self._is_article_relevant(article, company):
                            articles.append(article)
                            logger.info(f"Added backup article: {article.get('title', 'No title')}")
                        if len(articles) >= num_articles:
                            break
                    except Exception as e:
                        logger.warning(f"Backup article extraction error: {str(e)}")
                        continue
            except Exception as e:
                logger.warning(f"Backup article search failed: {str(e)}")
        
        # Last resort: if we still don't have enough articles, lower relevance threshold
        if len(articles) < num_articles:
            logger.info(f"Still only have {len(articles)} articles, lowering relevance threshold")
            try:
                for url in urls:
                    if url not in [a.get('url') for a in articles]:  # Skip already processed URLs
                        try:
                            article = self.extract_article_content(url)
                            if article:
                                # Less strict relevance check
                                if self._is_minimally_relevant(article, company):
                                    articles.append(article)
                                    logger.info(f"Added minimally relevant article: {article.get('title', 'No title')}")
                            if len(articles) >= num_articles:
                                break
                        except Exception as e:
                            continue
            except Exception as e:
                logger.warning(f"Lowered threshold search failed: {str(e)}")
        
        # Log final result
        logger.info(f"Found {len(articles)} relevant articles for {company}")
        
        # Return up to num_articles
        return articles[:num_articles]
    
    def _is_article_relevant(self, article: Dict[str, Any], company: str) -> bool:
        """Check if article is relevant to the company"""
        if not article or not article.get('content'):
            return False
            
        # Check if company name appears in title or first paragraph
        company_lower = company.lower()
        company_words = company_lower.split()
        
        # Check title
        if article.get('title'):
            title_lower = article['title'].lower()
            if company_lower in title_lower or any(word in title_lower for word in company_words if len(word) > 3):
                return True
        
        # Check first 500 chars of content
        content_start = article['content'][:500].lower()
        if company_lower in content_start or any(word in content_start for word in company_words if len(word) > 3):
            return True
            
        # More sophisticated check - count mentions
        content_lower = article['content'].lower()
        mention_count = content_lower.count(company_lower)
        
        # If company name has multiple words, check individual words
        for word in company_words:
            if len(word) > 3:  # Only check significant words
                mention_count += content_lower.count(word)
        
        # Article is relevant if company is mentioned multiple times
        return mention_count >= 2  # Require at least 2 mentions for relevance
    
    def _is_minimally_relevant(self, article: Dict[str, Any], company: str) -> bool:
        """Check if article is at least minimally relevant - used as fallback"""
        if not article or not article.get('content'):
            return False
            
        company_lower = company.lower()
        company_words = company_lower.split()
        
        # Check title
        if article.get('title'):
            title_lower = article['title'].lower()
            if company_lower in title_lower:
                return True
        
        # Check content for any mention
        content_lower = article['content'].lower()
        if company_lower in content_lower:
            return True
            
        # Check for industry keywords if company has more than one word
        if len(company_words) > 1:
            for word in company_words:
                if len(word) > 3 and content_lower.count(word) > 1:  # Significant word appears multiple times
                    return True
        
        return False
    
    def _get_backup_urls(self, company: str) -> List[str]:
        """Get backup URLs for when normal search fails"""
        backup_urls = []
        
        # Try to get Yahoo Finance company page
        ticker = self._guess_ticker(company)
        backup_urls.append(f"https://finance.yahoo.com/quote/{ticker}/news")
        
        # Try MarketWatch
        company_dash = company.replace(' ', '-').lower()
        backup_urls.append(f"https://www.marketwatch.com/investing/stock/{company_dash}/news")
        
        # Add Reuters
        company_plus = company.replace(' ', '+').lower()
        backup_urls.append(f"https://www.reuters.com/search/news?blob={company_plus}")
        
        # Add Bloomberg
        backup_urls.append(f"https://www.bloomberg.com/search?query={company_plus}")
        
        # Add Seeking Alpha
        backup_urls.append(f"https://seekingalpha.com/search?q={company_plus}")
        
        # Add CNBC
        backup_urls.append(f"https://www.cnbc.com/search/?query={company_plus}")
        
        # Add Investing.com
        backup_urls.append(f"https://www.investing.com/search/?q={company_plus}")
        
        # Add The Motley Fool
        backup_urls.append(f"https://www.fool.com/search/?q={company_plus}")
        
        return backup_urls


class SentimentAnalyzer:
    """Efficient sentiment analysis for news articles"""
    
    def __init__(self):
        try:
            self.sia = SentimentIntensityAnalyzer()
        except Exception as e:
            logger.warning(f"Error initializing SentimentIntensityAnalyzer: {str(e)}")
            self.sia = None
            
        try:
            self.stop_words = set(stopwords.words('english'))
        except:
            # Basic fallback stopwords
            self.stop_words = set(['i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'the', 'a', 'an', 'and', 'but', 'if', 'or'])
            
        try:
            self.lemmatizer = WordNetLemmatizer()
        except:
            # Simple fallback lemmatizer
            self.lemmatizer = type('', (), {'lemmatize': lambda self, word, pos='n': word})()

    def analyze_sentiment(self, text: str) -> Dict[str, Any]:
        """Analyze sentiment of news text"""
        if not text or len(text) < 100:  # Reduced minimum length
            return {"compound": 0, "pos": 0.33, "neu": 0.67, "neg": 0, "label": "neutral"}
            
        try:
            if self.sia:
                scores = self.sia.polarity_scores(text)
                label = 'neutral'
                if scores['compound'] >= 0.05:
                    label = 'positive'
                elif scores['compound'] <= -0.05:
                    label = 'negative'
                return {**scores, "label": label}
            else:
                # Simple fallback using keyword matching
                positive_words = ["good", "great", "excellent", "positive", "profit", "growth", "success", 
                                  "increase", "up", "higher", "beat", "strong", "gain", "improve", "opportunity"]
                negative_words = ["bad", "poor", "negative", "loss", "decline", "fail", "failure", "problem",
                                  "decrease", "down", "lower", "miss", "weak", "drop", "risk", "concern"]
                
                text_lower = text.lower()
                positive_count = sum(1 for word in positive_words if word in text_lower)
                negative_count = sum(1 for word in negative_words if word in text_lower)
                
                total = positive_count + negative_count or 1
                pos_score = positive_count / total
                neg_score = negative_count / total
                
                compound = (pos_score - neg_score) * 2
                compound = max(min(compound, 1.0), -1.0)
                
                label = 'neutral'
                if compound >= 0.05:
                    label = 'positive'
                elif compound <= -0.05:
                    label = 'negative'
                    
                return {
                    "compound": compound, 
                    "pos": pos_score, 
                    "neu": 1 - (pos_score + neg_score), 
                    "neg": neg_score, 
                    "label": label
                }
                
        except Exception as e:
            logger.warning(f"Sentiment analysis failed: {str(e)}")
            return {"compound": 0, "pos": 0.33, "neu": 0.67, "neg": 0, "label": "neutral"}

    def extract_topics(self, text: str, num_topics: int = 5) -> List[str]:
        """Extract key topics from news text"""
        if not text or len(text) < 100:  # Reduced minimum length
            return []
            
        try:
            # Tokenize and clean text
            tokens = word_tokenize(re.sub(r'[^\w\s]', '', text.lower()))
            
            # Filter stopwords and short words
            words = [self.lemmatizer.lemmatize(word) 
                    for word in tokens
                    if word not in self.stop_words and len(word) > 2]
            
            # Find most common words
            freq_dist = nltk.FreqDist(words)
            
            # Extract top words and handle bigrams
            top_words = [word for word, freq in freq_dist.most_common(num_topics*3) 
                         if not word.isdigit()]
            
            # Create bigrams for better topics
            bigrams = []
            for i in range(len(tokens) - 1):
                if (tokens[i] not in self.stop_words and 
                    tokens[i+1] not in self.stop_words and
                    len(tokens[i]) > 2 and 
                    len(tokens[i+1]) > 2):
                    bigram = f"{self.lemmatizer.lemmatize(tokens[i])} {self.lemmatizer.lemmatize(tokens[i+1])}"
                    bigrams.append(bigram)
            
            # Count bigrams
            bigram_freq = nltk.FreqDist(bigrams)
            top_bigrams = [bigram for bigram, freq in bigram_freq.most_common(num_topics*2)
                          if freq > 1]  # Only include repeated bigrams
            
            # Combine and filter results
            topics = top_bigrams + top_words
            
            # Business domain specific filter - add industry-specific terms
            business_topics = [
                "Market Share", "Revenue", "Growth", "Profit", "Earnings",
                "Quarterly Results", "Financial Performance", "Stock Price",
                "Investment", "Expansion", "New Product", "CEO", "Leadership",
                "Industry Trend", "Competition", "Merger", "Acquisition",
                "Strategy", "Innovation", "Technology", "Regulatory", "Market Trend"
            ]
            
            # Add relevant business topics that appear in the text
            for topic in business_topics:
                if topic.lower() in text.lower():
                    topics.append(topic)
            
            # Remove duplicates and return capitalized topics
            unique_topics = []
            seen = set()
            for topic in topics:
                if topic.lower() not in seen:
                    seen.add(topic.lower())
                    unique_topics.append(topic.title())
            
            return unique_topics[:num_topics]  # Return only top topics
                
        except Exception as e:
            logger.warning(f"Topic extraction failed: {str(e)}")
            return ["Business", "Finance", "Market"]

    def summarize_text(self, text: str, max_length: int = 250) -> str:
        """Create a concise summary of article text"""
        if not text or len(text) < 100:  # Reduced minimum length
            return ""
            
        try:
            # Simple extractive summarization using first sentences
            sentences = nltk.sent_tokenize(text)
            if len(sentences) <= 3:
                return text[:max_length] + "..." if len(text) > max_length else text
                
            # Use first 2-3 sentences as summary
            summary = ' '.join(sentences[:3])
            
            # Truncate if too long
            if len(summary) > max_length:
                summary = summary[:max_length] + "..."
                
            return summary
                
        except Exception as e:
            logger.warning(f"Summarization failed: {str(e)}")
            return text[:max_length] + "..." if len(text) > max_length else text


class ComparativeAnalyzer:
    """Comparative analysis for multiple news articles"""
    
    def __init__(self):
        pass

    def generate_comparative_report(self, company: str, articles: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Generate analysis report from news articles"""
        try:
            logger.info(f"Generating comparative report for {len(articles)} articles about {company}")
            
            # Extract sentiment data
            sentiment_data = [a.get('sentiment', {}) for a in articles if 'sentiment' in a]
            
            # Default sentiment if no data
            if not sentiment_data:
                sentiment_data = [{"compound": 0, "pos": 0.5, "neg": 0, "neu": 0.5, "label": "neutral"}]
            
            # Extract topics
            topics = [topic for a in articles for topic in a.get('topics', [])]
            if not topics:
                topics = ["Business", "Market", "Industry"]
            
            # Count sentiments
            sentiment_counts = {'positive': 0, 'negative': 0, 'neutral': 0}
            for sentiment in sentiment_data:
                label = sentiment.get('label', 'neutral')
                if label in sentiment_counts:
                    sentiment_counts[label] += 1
            
            # Calculate average sentiment score
            avg_compound = sum(s.get('compound', 0) for s in sentiment_data) / len(sentiment_data)
            
            # Group by source
            sources = {}
            for article in articles:
                source = article.get('source', 'unknown')
                if source not in sources:
                    sources[source] = []
                sources[source].append(article)
            
            # Find common and unique topics
            topic_counts = {}
            for topic in topics:
                if topic in topic_counts:
                    topic_counts[topic] += 1
                else:
                    topic_counts[topic] = 1
            
            common_topics = [topic for topic, count in topic_counts.items() 
                            if count > 1 and count >= len(articles) * 0.3]  # Topics in at least 30% of articles
            
            unique_topics = [topic for topic, count in topic_counts.items() 
                            if count == 1]  # Topics in only one article
            
            # Find coverage differences
            coverage_differences = []
            
            # Compare sentiment across sources
            if len(sources) > 1:
                source_sentiments = {}
                for source, source_articles in sources.items():
                    source_compounds = [a.get('sentiment', {}).get('compound', 0) for a in source_articles]
                    source_sentiments[source] = sum(source_compounds) / len(source_compounds) if source_compounds else 0
                
                # Find sources with significantly different sentiment
                for source1, score1 in source_sentiments.items():
                    for source2, score2 in source_sentiments.items():
                        if source1 != source2 and abs(score1 - score2) > 0.3:  # Significant difference threshold
                            sentiment1 = "positive" if score1 > 0 else "negative" if score1 < 0 else "neutral"
                            sentiment2 = "positive" if score2 > 0 else "negative" if score2 < 0 else "neutral"
                            
                            if sentiment1 != sentiment2:
                                coverage_differences.append({
                                    "comparison": f"{source1} coverage is more {sentiment1} than {source2}",
                                    "impact": f"Consider the reporting bias - {source1} has sentiment score {score1:.2f} vs {source2} at {score2:.2f}"
                                })
            
            # Generate final sentiment analysis
            final_sentiment = "neutral"
            if avg_compound >= 0.05:
                final_sentiment = "positive"
            elif avg_compound <= -0.05:
                final_sentiment = "negative"
            
            # More nuanced sentiment description
            if avg_compound >= 0.5:
                final_sentiment_analysis = f"Overall extremely positive coverage for {company}"
            elif avg_compound >= 0.2:
                final_sentiment_analysis = f"Overall positive coverage for {company}"
            elif avg_compound >= 0.05:
                final_sentiment_analysis = f"Overall slightly positive coverage for {company}"
            elif avg_compound > -0.05:
                final_sentiment_analysis = f"Overall neutral coverage for {company}"
            elif avg_compound > -0.2:
                final_sentiment_analysis = f"Overall slightly negative coverage for {company}"
            elif avg_compound > -0.5:
                final_sentiment_analysis = f"Overall negative coverage for {company}"
            else:
                final_sentiment_analysis = f"Overall extremely negative coverage for {company}"
            
            # Add sentiment distribution to analysis
            pos_count = sentiment_counts.get('positive', 0)
            neg_count = sentiment_counts.get('negative', 0)
            neu_count = sentiment_counts.get('neutral', 0)
            total_count = pos_count + neg_count + neu_count
            
            if total_count > 0:
                pos_pct = (pos_count / total_count) * 100
                neg_pct = (neg_count / total_count) * 100
                neu_pct = (neu_count / total_count) * 100
                
                final_sentiment_analysis += f". Distribution: {pos_pct:.1f}% positive, {neg_pct:.1f}% negative, {neu_pct:.1f}% neutral articles."
            
            # Generate time-based analysis if timestamp data is available
            time_analysis = {}
            articles_with_dates = [a for a in articles if 'publish_date' in a and a['publish_date'] != "Unknown"]
            
            if articles_with_dates:
                # Sort articles by date
                try:
                    from datetime import datetime
                    
                    for article in articles_with_dates:
                        try:
                            date_str = article['publish_date']
                            date_obj = datetime.strptime(date_str, '%Y-%m-%d')
                            article['date_obj'] = date_obj
                        except:
                            pass
                    
                    articles_with_dates = [a for a in articles_with_dates if 'date_obj' in a]
                    articles_with_dates.sort(key=lambda x: x['date_obj'])
                    
                    # Compare sentiment over time
                    if len(articles_with_dates) >= 2:
                        older_articles = articles_with_dates[:len(articles_with_dates)//2]
                        newer_articles = articles_with_dates[len(articles_with_dates)//2:]
                        
                        older_sentiment = sum(a.get('sentiment', {}).get('compound', 0) for a in older_articles) / len(older_articles)
                        newer_sentiment = sum(a.get('sentiment', {}).get('compound', 0) for a in newer_articles) / len(newer_articles)
                        
                        if abs(newer_sentiment - older_sentiment) > 0.2:
                            sentiment_direction = "improving" if newer_sentiment > older_sentiment else "declining"
                            time_analysis["sentiment_trend"] = f"Media sentiment appears to be {sentiment_direction} over time"
                except:
                    # Skip temporal analysis if there are issues
                    pass
            
            # Create final report
            report = {
                "company": company,
                "articles": articles,
                "sentiment_counts": sentiment_counts,
                "final_sentiment": final_sentiment,
                "final_sentiment_analysis": final_sentiment_analysis,
                "avg_compound_score": avg_compound,
                "comparative_sentiment_score": {
                    "topic_overlap": {
                        "common_topics": common_topics,
                        "unique_topics": unique_topics
                    },
                    "coverage_differences": coverage_differences
                }
            }
            
            # Add time analysis if available
            if time_analysis:
                report["time_analysis"] = time_analysis
            
            logger.info(f"Generated comparative report with {len(articles)} articles and {len(common_topics)} common topics")
            return report
            
        except Exception as e:
            logger.error(f"Error generating comparative report: {str(e)}")
            # Return basic report on error
            return {
                "company": company,
                "articles": articles,
                "sentiment_counts": {'positive': 0, 'negative': 0, 'neutral': len(articles)},
                "final_sentiment": "neutral",
                "final_sentiment_analysis": f"Unable to determine sentiment for {company} due to analysis error",
                "avg_compound_score": 0,
                "comparative_sentiment_score": {
                    "topic_overlap": {
                        "common_topics": [],
                        "unique_topics": []
                    },
                    "coverage_differences": []
                }
            }
            
    def find_significant_differences(self, articles: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Find significant differences between articles"""
        differences = []
        
        try:
            # Group articles by sentiment
            positive_articles = [a for a in articles if a.get('sentiment', {}).get('label', 'neutral') == 'positive']
            negative_articles = [a for a in articles if a.get('sentiment', {}).get('label', 'neutral') == 'negative']
            
            # Compare topics between positive and negative articles
            if positive_articles and negative_articles:
                positive_topics = set()
                for article in positive_articles:
                    for topic in article.get('topics', []):
                        positive_topics.add(topic.lower())
                
                negative_topics = set()
                for article in negative_articles:
                    for topic in article.get('topics', []):
                        negative_topics.add(topic.lower())
                
                # Find topics that appear predominantly in positive or negative articles
                pos_only_topics = positive_topics - negative_topics
                neg_only_topics = negative_topics - positive_topics
                
                if pos_only_topics:
                    differences.append({
                        "type": "topic_sentiment",
                        "description": f"Topics only in positive articles: {', '.join(pos_only_topics)}",
                        "impact": "These topics are associated with positive sentiment"
                    })
                
                if neg_only_topics:
                    differences.append({
                        "type": "topic_sentiment",
                        "description": f"Topics only in negative articles: {', '.join(neg_only_topics)}",
                        "impact": "These topics are associated with negative sentiment"
                    })
            
            # Compare sources for bias
            sources = {}
            for article in articles:
                source = article.get('source', 'unknown')
                if source not in sources:
                    sources[source] = []
                sources[source].append(article)
            
            if len(sources) > 1:
                for source, source_articles in sources.items():
                    if len(source_articles) >= 2:
                        source_sentiment = sum(a.get('sentiment', {}).get('compound', 0) for a in source_articles) / len(source_articles)
                        other_articles = [a for a in articles if a.get('source', 'unknown') != source]
                        
                        if other_articles:
                            other_sentiment = sum(a.get('sentiment', {}).get('compound', 0) for a in other_articles) / len(other_articles)
                            
                            if abs(source_sentiment - other_sentiment) > 0.3:
                                bias_direction = "positive" if source_sentiment > other_sentiment else "negative"
                                differences.append({
                                    "type": "source_bias",
                                    "description": f"{source} shows {bias_direction} bias compared to other sources",
                                    "impact": f"{source} sentiment score: {source_sentiment:.2f}, Others: {other_sentiment:.2f}"
                                })
            
            return differences
            
        except Exception as e:
            logger.error(f"Error finding significant differences: {str(e)}")
            return []
            
    def generate_executive_summary(self, report: Dict[str, Any]) -> str:
        """Generate an executive summary of the analysis"""
        try:
            company = report.get('company', 'Unknown company')
            sentiment = report.get('final_sentiment', 'neutral')
            sentiment_analysis = report.get('final_sentiment_analysis', '')
            
            # Get sentiment counts
            sentiment_counts = report.get('sentiment_counts', {})
            pos_count = sentiment_counts.get('positive', 0)
            neg_count = sentiment_counts.get('negative', 0)
            neu_count = sentiment_counts.get('neutral', 0)
            total_count = pos_count + neg_count + neu_count
            
            # Get topics
            common_topics = report.get('comparative_sentiment_score', {}).get('topic_overlap', {}).get('common_topics', [])
            
            # Build summary
            summary = f"Executive Summary for {company}:\n\n"
            
            # Add sentiment summary
            summary += f"Overall sentiment: {sentiment.upper()}\n"
            summary += f"{sentiment_analysis}\n\n"
            
            # Add article count
            summary += f"Analysis based on {total_count} articles"
            if total_count > 0:
                summary += f": {pos_count} positive, {neg_count} negative, {neu_count} neutral.\n\n"
            else:
                summary += ".\n\n"
            
            # Add top topics
            if common_topics:
                summary += f"Top topics: {', '.join(common_topics[:5])}\n\n"
            
            # Add time analysis if available
            if 'time_analysis' in report and 'sentiment_trend' in report['time_analysis']:
                summary += f"Trend: {report['time_analysis']['sentiment_trend']}\n\n"
            
            # Add coverage differences
            differences = report.get('comparative_sentiment_score', {}).get('coverage_differences', [])
            if differences:
                summary += "Key differences in coverage:\n"
                for diff in differences[:3]:  # Show top 3 differences
                    summary += f"- {diff.get('comparison', '')}\n"
                summary += "\n"
            
            return summary
            
        except Exception as e:
            logger.error(f"Error generating executive summary: {str(e)}")
            return f"Executive Summary for {report.get('company', 'Unknown company')}: Unable to generate detailed summary due to error."
