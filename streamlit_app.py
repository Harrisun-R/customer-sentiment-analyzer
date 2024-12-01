import streamlit as st
import pandas as pd
import nltk
from textblob import TextBlob
import plotly.express as px
import plotly.graph_objs as go
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import io

# Download necessary NLTK resources
nltk.download('punkt', quiet=True)

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

def main():
    st.title('ð Product Customer Sentiment Analyzer')
    st.markdown('Analyze customer feedback and extract meaningful insights!')

    # Sidebar for input methods
    input_method = st.sidebar.radio('Select Input Method', 
                                    ['Text Input', 'CSV Upload'])

    # Sentiment Analysis Container
    with st.container():
        if input_method == 'Text Input':
            # Text Input Section
            review_text = st.text_area('Enter Customer Reviews (one per line)', height=200)
            
            if st.button('Analyze Sentiment'):
                if review_text:
                    # Process multiple reviews
                    reviews = review_text.split('\n')
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
                    
                    # Word Cloud
                    st.subheader('Word Cloud of Reviews')
                    word_cloud_img = generate_word_cloud(review_text)
                    st.image(word_cloud_img)

        else:
            # CSV Upload Section
            uploaded_file = st.file_uploader('Upload CSV with Reviews', type=['csv'])
            
            if uploaded_file is not None:
                try:
                    df = pd.read_csv(uploaded_file)
                    
                    # Assume a column named 'review' or prompt user to select
                    review_column = st.selectbox('Select Review Column', df.columns)
                    
                    if st.button('Analyze CSV'):
                        # Perform sentiment analysis on selected column
                        df['Sentiment'], df['Polarity'] = zip(*df[review_column].apply(perform_sentiment_analysis))
                        
                        # Sentiment Distribution
                        st.subheader('Sentiment Distribution')
                        sentiment_counts = df['Sentiment'].value_counts()
                        fig_sentiment = px.pie(values=sentiment_counts.values, 
                                               names=sentiment_counts.index, 
                                               title='Sentiment Breakdown')
                        st.plotly_chart(fig_sentiment)
                        
                        # Word Cloud
                        st.subheader('Word Cloud of Reviews')
                        word_cloud_img = generate_word_cloud(' '.join(df[review_column]))
                        st.image(word_cloud_img)
                        
                        # Detailed Sentiment Analysis Table
                        st.subheader('Detailed Sentiment Analysis')
                        st.dataframe(df[['Sentiment', 'Polarity', review_column]])
                
                except Exception as e:
                    st.error(f"Error processing CSV: {e}")

    # Footer
    st.markdown('---')
    st.markdown('**Powered by AI-driven Sentiment Analysis**')

if __name__ == '__main__':
    main()
