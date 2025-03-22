# News Summarization and Text-to-Speech Application

This project is a web-based application designed to extract key details from multiple news articles about a specified company, perform sentiment analysis, conduct comparative analysis, and generate text-to-speech (TTS) output in Hindi. Users can input a company name and receive a structured sentiment report along with an audio summary in Hindi.&#8203;:contentReference[oaicite:0]{index=0}

## Files Overview

- **app.py**: The main application script that integrates various components to run the News Summarizer.​

- **requirements.txt**: Lists the Python dependencies required to run the application.​

- **tts.py**: Handles text-to-speech functionalities, converting summarized text into spoken words.​

- **utils.py**: Contains utility functions supporting the application's core features.

## Features

- **News Extraction**: retrieves structured information from unstructured news articles.

- **Sentiment Analysis**: determines the emotional tone—positive, negative, or neutral—within the text.

- **Comparative Analysis**: Provides insights into how the company's news coverage varies across different articles.
- 
- **Text-to-Speech (TTS)**: Converts the summarized content into Hindi speech using an open-source TTS model.

- **User Interface**: Offers a simple web-based interface using Streamlit. Users can input a company name to fetch news articles and generate the sentiment report.

## SETUP Installation

 **Clone the repository** : 
   git clone https://github.com/Subhasya/news-summarizer.git
   cd news-summarizer

**Install dependencies**: pip install -r requirements.txt

**Run the application**: streamlit run app.py.

## API Endpoints

/fetch_news, /analyze_sentiment, /generate_tts

## Deployment
The application is deployed on Hugging Face Spaces and can be accessed here [https://huggingface.co/spaces/Subhasya/News_Summarizer].

## Usuage
- Open the web application in your browser.​

- Enter the company name in the input field.​

- Click the "Submit" button to fetch news articles and generate the sentiment report.​

- Review the structured report, which includes article titles, summaries, sentiments, topics, comparative analysis, and a playable Hindi audio summary.



