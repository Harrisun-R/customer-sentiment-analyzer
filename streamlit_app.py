import streamlit as st
import pandas as pd
import nltk
from textblob import TextBlob
import plotly.express as px
import plotly.graph_objs as go
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import io
from sklearn.feature_extraction.text import CountVectorizer
from nltk.corpus import stopwords
import string

# Download necessary NLTK resources
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)

# Sentiment analysis function
def perform_sentiment_analysis(text):
    """
    Perform sentiment analysis on given text
    Returns sentiment label and polarity score
    """
    blob = TextBlob(text)
    polarity = blob.sentiment.polarity
    
    if polarity > 0:
        sentiment = 'Positive'
    elif polarity < 0:
        sentiment = 'Negative'
    else:
        sentiment = 'Neutral'
    
    return sentiment, polarity

# Text preprocessing function
def preprocess_text(text, remove_stopwords, remove_punctuation, to_lowercase):
    """
    Preprocess the input text based on user-selected options.
    """
    if to_lowercase:
        text = text.lower()
    
    if remove_punctuation:
        text = text.translate(str.maketrans('', '', string.punctuation))
    
    if remove_stopwords:
        stop_words = set(stopwords.words('english'))
        text = ' '.join([word for word in text.split() if word not in stop_words])
    
    return text

# Generate word cloud
def generate_word_cloud(text):
    """
    Generate word cloud from text
    """
    wordcloud = WordCloud(width=800, height=400, 
                          background_color='white').generate(text)
    
    plt.figure(figsize=(10,5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    
    # Save to a bytes buffer
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    return buf

# Extract top keywords using CountVectorizer
def extract_keywords(text, n_keywords=10):
    """
    Extract top keywords using TF-IDF
    """
    vectorizer = CountVectorizer(stop_words='english', ngram_range=(1, 2), max_features=n_keywords)
    X = vectorizer.fit_transform([text])
    keywords = vectorizer.get_feature_names_out()
    return keywords
Name = "Harrisun Raj Mohan"
URL = "https://www.linkedin.com/in/your-profile/harrisun-raj-mohan"
def main():
    st.title('🔍 Product Customer Sentiment Analyzer')
    st.markdown('Analyze customer feedback and extract meaningful insights!')
    st.write(f"Created by {Name}. [Connect on LinkedIn]({URL})")

    # Sidebar for input methods
    input_method = st.sidebar.radio('Select Input Method', 
                                    ['Text Input', 'CSV Upload', 'Comparative Analysis'])

    # Text Preprocessing options
    st.sidebar.header('Text Preprocessing Options')
    remove_stopwords = st.sidebar.checkbox('Remove Stopwords')
    remove_punctuation = st.sidebar.checkbox('Remove Punctuation')
    to_lowercase = st.sidebar.checkbox('Convert to Lowercase')
    
    # Sentiment Analysis Container
    with st.container():
        if input_method == 'Text Input':
            # Text Input Section
            review_text = st.text_area('Enter Customer Reviews (one per line)', height=200)
            
            if st.button('Analyze Sentiment'):
                if review_text:
                    # Preprocess reviews based on user selections
                    reviews = review_text.split('\n')
                    reviews = [preprocess_text(review, remove_stopwords, remove_punctuation, to_lowercase) for review in reviews]
                    
                    sentiment_results = [perform_sentiment_analysis(review) for review in reviews if review.strip()]
                    
                    # Create DataFrame
                    df = pd.DataFrame(sentiment_results, columns=['Sentiment', 'Polarity'])
                    
                    # Sentiment Distribution
                    st.subheader('Sentiment Distribution')
                    sentiment_counts = df['Sentiment'].value_counts()
                    fig_sentiment = px.pie(values=sentiment_counts.values, 
                                           names=sentiment_counts.index, 
                                           title='Sentiment Breakdown')
                    st.plotly_chart(fig_sentiment)

                    # Sentiment Summary
                    sentiment_summary = df['Sentiment'].value_counts(normalize=True) * 100
                    st.subheader('Sentiment Summary')
                    st.write(f"Positive: {sentiment_summary.get('Positive', 0):.2f}%")
                    st.write(f"Negative: {sentiment_summary.get('Negative', 0):.2f}%")
                    st.write(f"Neutral: {sentiment_summary.get('Neutral', 0):.2f}%")
                    
                    # Keyword Extraction
                    st.subheader('Top Keywords')
                    keywords = extract_keywords(' '.join(reviews))
                    st.write(keywords)
                    
                    # Word Cloud
                    st.subheader('Word Cloud of Reviews')
                    word_cloud_img = generate_word_cloud(review_text)
                    st.image(word_cloud_img)

                    # Sentiment Intensity Visualization
                    st.subheader('Sentiment Intensity')
                    fig_intensity = px.histogram(df, x='Polarity', nbins=30, title='Polarity Intensity Distribution')
                    st.plotly_chart(fig_intensity)
                    
                    # Export CSV option
                    st.subheader('Export Analysis')
                    csv = df.to_csv(index=False).encode()
                    st.download_button('Download CSV', csv, file_name='sentiment_analysis.csv')

        elif input_method == 'CSV Upload':
            # CSV Upload Section
            uploaded_file = st.file_uploader('Upload CSV with Reviews', type=['csv'])
            
            if uploaded_file is not None:
                try:
                    df = pd.read_csv(uploaded_file)
                    
                    # Assume a column named 'review' or prompt user to select
                    review_column = st.selectbox('Select Review Column', df.columns)
                    
                    if st.button('Analyze CSV'):
                        # Preprocess the reviews
                        df[review_column] = df[review_column].apply(lambda x: preprocess_text(x, remove_stopwords, remove_punctuation, to_lowercase))
                        df['Sentiment'], df['Polarity'] = zip(*df[review_column].apply(perform_sentiment_analysis))
                        
                        # Sentiment Distribution
                        st.subheader('Sentiment Distribution')
                        sentiment_counts = df['Sentiment'].value_counts()
                        fig_sentiment = px.pie(values=sentiment_counts.values, 
                                               names=sentiment_counts.index, 
                                               title='Sentiment Breakdown')
                        st.plotly_chart(fig_sentiment)
                        
                        # Sentiment Summary
                        sentiment_summary = df['Sentiment'].value_counts(normalize=True) * 100
                        st.subheader('Sentiment Summary')
                        st.write(f"Positive: {sentiment_summary.get('Positive', 0):.2f}%")
                        st.write(f"Negative: {sentiment_summary.get('Negative', 0):.2f}%")
                        st.write(f"Neutral: {sentiment_summary.get('Neutral', 0):.2f}%")
                        
                        # Keyword Extraction
                        st.subheader('Top Keywords')
                        keywords = extract_keywords(' '.join(df[review_column]))
                        st.write(keywords)
                        
                        # Word Cloud
                        st.subheader('Word Cloud of Reviews')
                        word_cloud_img = generate_word_cloud(' '.join(df[review_column]))
                        st.image(word_cloud_img)

                        # Sentiment Intensity Visualization
                        st.subheader('Sentiment Intensity')
                        fig_intensity = px.histogram(df, x='Polarity', nbins=30, title='Polarity Intensity Distribution')
                        st.plotly_chart(fig_intensity)

                        # Export CSV option
                        st.subheader('Export Analysis')
                        csv = df.to_csv(index=False).encode()
                        st.download_button('Download CSV', csv, file_name='sentiment_analysis.csv')
                
                except Exception as e:
                    st.error(f"Error processing CSV: {e}")

        elif input_method == 'Comparative Analysis':
            # Comparative Analysis Section
            st.subheader('Upload Reviews Before and After Product Update')

            # Upload CSV before product update
            uploaded_file_before = st.file_uploader('Upload CSV for Reviews Before Product Update', type=['csv'], key='before')
            
            # Upload CSV after product update
            uploaded_file_after = st.file_uploader('Upload CSV for Reviews After Product Update', type=['csv'], key='after')
            
            if uploaded_file_before and uploaded_file_after:
                try:
                    df_before = pd.read_csv(uploaded_file_before)
                    df_after = pd.read_csv(uploaded_file_after)
                    
                    review_column_before = st.selectbox('Select Review Column for Before Update', df_before.columns, key='review_before')
                    review_column_after = st.selectbox('Select Review Column for After Update', df_after.columns, key='review_after')
                    
                    # Analyze Before Update
                    df_before[review_column_before] = df_before[review_column_before].apply(lambda x: preprocess_text(x, remove_stopwords, remove_punctuation, to_lowercase))
                    df_before['Sentiment'], df_before['Polarity'] = zip(*df_before[review_column_before].apply(perform_sentiment_analysis))
                    
                    # Analyze After Update
                    df_after[review_column_after] = df_after[review_column_after].apply(lambda x: preprocess_text(x, remove_stopwords, remove_punctuation, to_lowercase))
                    df_after['Sentiment'], df_after['Polarity'] = zip(*df_after[review_column_after].apply(perform_sentiment_analysis))
                    
                    # Comparative Sentiment Distribution
                    st.subheader('Comparative Sentiment Distribution')
                    fig_before = px.pie(df_before['Sentiment'].value_counts().reset_index(), 
                                        names='index', values='Sentiment', title='Before Product Update')
                    st.plotly_chart(fig_before)
                    
                    fig_after = px.pie(df_after['Sentiment'].value_counts().reset_index(), 
                                       names='index', values='Sentiment', title='After Product Update')
                    st.plotly_chart(fig_after)
                    
                    # Export CSV options
                    st.subheader('Export Analysis for Both Datasets')
                    csv_before = df_before.to_csv(index=False).encode()
                    csv_after = df_after.to_csv(index=False).encode()

                    st.download_button('Download CSV for Before Update', csv_before, file_name='sentiment_analysis_before.csv')
                    st.download_button('Download CSV for After Update', csv_after, file_name='sentiment_analysis_after.csv')

                except Exception as e:
                    st.error(f"Error processing Comparative Analysis: {e}")

    # Footer
    st.markdown('---')
    st.markdown('**Powered by AI-driven Sentiment Analysis**')

if __name__ == '__main__':
    main()
