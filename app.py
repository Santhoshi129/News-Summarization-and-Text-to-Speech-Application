import streamlit as st
import base64
import pandas as pd
import time
import os
import nltk
import traceback

# Download NLTK data files
try:
    nltk.download('punkt', quiet=True)
    nltk.download('stopwords', quiet=True)
    nltk.download('wordnet', quiet=True)
    nltk.download('vader_lexicon', quiet=True)
except Exception as e:
    st.warning(f"NLTK download issue: {str(e)}")

# Import our modules directly
from utils import NewsExtractor, SentimentAnalyzer, ComparativeAnalyzer
from tts import TextToSpeechConverter

# Initialize our components
news_extractor = NewsExtractor()
sentiment_analyzer = SentimentAnalyzer()
comparative_analyzer = ComparativeAnalyzer()
tts_converter = TextToSpeechConverter()

st.set_page_config(
    page_title="News Summarization & Sentiment Analysis",
    page_icon="📰",
    layout="wide"
)

def get_news(company):
    """Get news articles for a company."""
    try:
        st.write(f"Finding news articles for {company}...")
        
        # Extract articles
        articles = news_extractor.get_company_news(company)
        
        # Handle empty results
        if not articles:
            st.error(f"No news found for '{company}'. Try a different company.")
            return None
        
        st.write(f"Found {len(articles)} articles. Analyzing content...")
        
        # Process each article
        processed_articles = []
        for article in articles:
            if 'content' in article and article['content']:
                # Analyze sentiment
                sentiment = sentiment_analyzer.analyze_sentiment(article['content'])
                
                # Extract topics
                topics = sentiment_analyzer.extract_topics(article['content'])
                
                # Ensure summary exists
                if 'summary' not in article or not article['summary']:
                    article['summary'] = sentiment_analyzer.summarize_text(article['content'])
                
                # Add to processed articles
                processed_article = {
                    "title": article.get('title', 'No title'),
                    "summary": article.get('summary', 'No summary available'),
                    "content": article.get('content', ''),
                    "url": article.get('url', ''),
                    "sentiment": sentiment,
                    "topics": topics
                }
                processed_articles.append(processed_article)
        
        if not processed_articles:
            st.warning("Could not process any articles. Using default data.")
            # Use the first article as fallback
            for article in articles:
                processed_articles.append({
                    "title": article.get('title', 'No title'),
                    "summary": article.get('summary', 'No summary available'),
                    "content": article.get('content', ''),
                    "url": article.get('url', ''),
                    "sentiment": {"compound": 0, "pos": 0.5, "neg": 0, "neu": 0.5, "label": "neutral"},
                    "topics": ["Business", "Market", "Industry"]
                })
        
        # Generate comparative report
        st.write("Generating comparative report...")
        report = comparative_analyzer.generate_comparative_report(company, processed_articles)
        
        # Generate TTS
        st.write("Generating speech...")
        tts_result = tts_converter.generate_summary_speech(report)
        
        if tts_result['success'] and tts_result['output_file']:
            # Read the audio file and encode to base64
            with open(tts_result['output_file'], 'rb') as audio_file:
                audio_data = base64.b64encode(audio_file.read()).decode('utf-8')
            
            report['audio_data'] = audio_data
            report['hindi_text'] = tts_result['hindi_text']
        
        return report
    except Exception as e:
        st.error(f"Error processing news: {str(e)}")
        st.error(f"Traceback: {traceback.format_exc()}")
        return None

def display_sentiment_chart(sentiment_counts):
    """Display a chart of sentiment distribution."""
    if not sentiment_counts:
        return
    
    # Create data for chart
    chart_data = pd.DataFrame({
        'Sentiment': ['Positive', 'Negative', 'Neutral'],
        'Count': [
            sentiment_counts.get('positive', 0),
            sentiment_counts.get('negative', 0),
            sentiment_counts.get('neutral', 0)
        ]
    })
    
    # Display chart
    st.bar_chart(chart_data.set_index('Sentiment'))

def play_audio(audio_data):
    """Play audio from base64 data."""
    try:
        audio_bytes = base64.b64decode(audio_data)
        st.audio(audio_bytes, format='audio/mp3')
    except Exception as e:
        st.error(f"Error playing audio: {str(e)}")

def main():
    """Main function for Streamlit app."""
    st.title("News Summarization & Sentiment Analysis")
    
    # Sidebar for company input
    st.sidebar.header("Input Options")
    
    # Default company options or user input
    company_options = ["Tesla", "Apple", "Google", "Microsoft", "Amazon", "Other"]
    company_choice = st.sidebar.selectbox("Select Company", company_options)
    
    if company_choice == "Other":
        company = st.sidebar.text_input("Enter Company Name")
    else:
        company = company_choice
    
    # Analysis button
    if st.sidebar.button("Analyze News") and company:
        with st.spinner(f'Fetching and analyzing news for {company}...'):
            # Show progress bar
            progress_bar = st.progress(0)
            for i in range(100):
                # Update progress bar
                progress_bar.progress(i + 1)
                time.sleep(0.01)
            
            # Get news data
            data = get_news(company)
            
            # Display results
            if data and 'articles' in data:
                st.success("Analysis Complete!")
                
                # Display sentiment summary
                st.header("Sentiment Analysis Summary")
                st.subheader(data.get('final_sentiment_analysis', 'No sentiment analysis available'))
                
                # Display audio player if available
                if 'audio_data' in data:
                    st.subheader("Audio Summary (Hindi)")
                    play_audio(data['audio_data'])
                    st.caption("Hindi Text:")
                    st.text(data.get('hindi_text', ''))
                
                # Display charts
                st.header("Sentiment Distribution")
                if 'sentiment_counts' in data:
                    display_sentiment_chart(data['sentiment_counts'])
                
                # Display articles
                st.header("News Articles")
                for idx, article in enumerate(data.get('articles', [])):
                    with st.expander(f"{idx+1}. {article.get('title', 'No Title')}"):
                        # Show sentiment badge
                        sentiment = article.get('sentiment', {}).get('label', 'neutral')
                        if sentiment == 'positive':
                            st.success(f"Sentiment: {sentiment.upper()}")
                        elif sentiment == 'negative':
                            st.error(f"Sentiment: {sentiment.upper()}")
                        else:
                            st.info(f"Sentiment: {sentiment.upper()}")
                        
                        # Show summary
                        st.subheader("Summary")
                        st.write(article.get('summary', 'No summary available'))
                        
                        # Show topics
                        if 'topics' in article and article['topics']:
                            st.subheader("Topics")
                            st.write(", ".join(article['topics']))
                        
                        # Show URL
                        st.subheader("Source")
                        st.write(f"[Read full article]({article.get('url', '#')})")
                
                # Display comparative analysis
                st.header("Comparative Analysis")
                if 'comparative_sentiment_score' in data and 'topic_overlap' in data['comparative_sentiment_score']:
                    topic_overlap = data['comparative_sentiment_score']['topic_overlap']
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        st.subheader("Common Topics")
                        if topic_overlap['common_topics']:
                            st.write(", ".join(topic_overlap['common_topics']))
                        else:
                            st.write("No common topics found")
                    
                    with col2:
                        st.subheader("Unique Topics")
                        if topic_overlap['unique_topics']:
                            st.write(", ".join(topic_overlap['unique_topics'][:5]))
                        else:
                            st.write("No unique topics found")
                
                # Display coverage differences
                if 'comparative_sentiment_score' in data and 'coverage_differences' in data['comparative_sentiment_score']:
                    coverage_diffs = data['comparative_sentiment_score']['coverage_differences']
                    
                    if coverage_diffs:
                        st.subheader("Coverage Differences")
                        for diff in coverage_diffs:
                            st.write(f"**Comparison:** {diff.get('comparison', '')}")
                            st.write(f"**Impact:** {diff.get('impact', '')}")
            else:
                st.error("No data available or error in analysis")

if __name__ == "__main__":
    main()