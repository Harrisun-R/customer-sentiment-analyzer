# Product Customer Sentiment Analyzer

## Overview
This Streamlit application provides AI-powered customer sentiment analysis for product managers and professionals.

## Features
- Analyze customer reviews in real-time
- Upload CSV files for bulk sentiment analysis
- Visualize sentiment distribution
- Generate word cloud for frequently mentioned topics

## Setup and Installation

### Prerequisites
- Python 3.10+
- pip package manager

### Installation Steps
1. Clone the repository
2. Create a virtual environment
```bash
python -m venv venv
source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
```

3. Install dependencies
```bash
pip install -r requirements.txt
```

4. Download NLTK resources
```bash
python -c "import nltk; nltk.download('punkt')"
```

### Running the Application
```bash
streamlit run sentiment_analyzer.py
```

## Usage
- Input reviews directly in the text area
- Upload a CSV file with reviews
- Select the review column from your CSV
- Click 'Analyze' to generate insights

## Libraries Used
- Streamlit
- NLTK
- TextBlob
- Plotly
- WordCloud

## Contributing
Pull requests are welcome. For major changes, please open an issue first.
