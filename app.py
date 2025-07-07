import streamlit as st
import pandas as pd
import plotly.express as px
from wordcloud import WordCloud, STOPWORDS
import matplotlib.pyplot as plt
from textblob import TextBlob
from datetime import datetime

# ==================== CONFIGURATION ====================
st.set_page_config(
    page_title="Airline Sentiment Dashboard",
    page_icon="✈️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==================== SESSION STATE ====================
if 'selected_airline' not in st.session_state:
    st.session_state.selected_airline = "All"
if 'date_range' not in st.session_state:
    st.session_state.date_range = None
if 'sentiment_filter' not in st.session_state:
    st.session_state.sentiment_filter = ["positive", "neutral", "negative"]

# ==================== DATA LOADING ====================
@st.cache_data(ttl=3600)
def load_data():
    try:
        data = pd.read_csv("Tweets.csv")
        data['tweet_created'] = pd.to_datetime(data['tweet_created'])
        return data
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return pd.DataFrame()

data = load_data()

# ==================== UTILITY FUNCTIONS ====================
def analyze_sentiment(text):
    """Analyze sentiment using TextBlob"""
    analysis = TextBlob(text)
    polarity = analysis.sentiment.polarity
    if polarity > 0.1:
        return "positive", polarity
    elif polarity < -0.1:
        return "negative", polarity
    else:
        return "neutral", polarity

def generate_wordcloud(text, sentiment):
    """Generate a word cloud for the given text"""
    if not text or text.isspace():
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.text(0.5, 0.5, "No text data available", ha='center', va='center', color='white')
        ax.axis('off')
        return fig
    
    colormap = {
        'positive': 'viridis',
        'neutral': 'plasma',
        'negative': 'Reds'
    }.get(sentiment, 'viridis')
    
    wc = WordCloud(
        width=800,
        height=400,
        background_color='#1a1a1a',
        colormap=colormap,
        stopwords=STOPWORDS,
        max_words=200
    ).generate(text)
    
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.imshow(wc, interpolation='bilinear')
    ax.axis('off')
    return fig

# ==================== DARK THEME STYLING ====================
def apply_dark_theme():
    st.markdown("""
    <style>
    :root {
        --primary: #4361ee;
        --secondary: #3f37c9;
        --accent: #4895ef;
    }
    
    .stApp {
        background-color: #0e1117;
        color: white;
    }
    
    [data-testid="stSidebar"] {
        background: rgba(30, 30, 30, 0.8) !important;
        backdrop-filter: blur(12px);
    }
    
    .metric-card {
        background: rgba(30, 30, 30, 0.7);
        border-radius: 10px;
        padding: 15px;
        margin: 5px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    
    .stButton>button {
        background: linear-gradient(135deg, #4361ee 0%, #4895ef 100%);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 10px 20px;
        font-weight: 500;
    }
    
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 5px 15px rgba(67, 97, 238, 0.4);
    }
    
    .sentiment-box {
        border-radius: 10px;
        padding: 15px;
        margin: 10px 0;
    }
    
    .positive {
        background: rgba(16, 185, 129, 0.2);
        border-left: 4px solid #10b981;
    }
    
    .neutral {
        background: rgba(245, 158, 11, 0.2);
        border-left: 4px solid #f59e0b;
    }
    
    .negative {
        background: rgba(239, 68, 68, 0.2);
        border-left: 4px solid #ef4444;
    }
    </style>
    """, unsafe_allow_html=True)

# ==================== SIDEBAR ====================
def render_sidebar():
    with st.sidebar:
        st.title("✈️ Airline Analytics")
        st.markdown("---")
        
        # Airline Filter
        airlines = ["All"] + sorted(data['airline'].unique().tolist())
        st.session_state.selected_airline = st.selectbox(
            "Select Airline",
            airlines,
            index=0
        )
        
        # Date Range Filter
        min_date = data['tweet_created'].min().date()
        max_date = data['tweet_created'].max().date()
        st.session_state.date_range = st.date_input(
            "Date Range",
            [min_date, max_date]
        )
        
        # Sentiment Filter
        st.session_state.sentiment_filter = st.multiselect(
            "Filter Sentiments",
            ["positive", "neutral", "negative"],
            default=["positive", "neutral", "negative"]
        )
        
        st.markdown("---")
        st.caption("© 2023 Airline Sentiment Dashboard")

# ==================== MAIN CONTENT ====================
def render_main_content():
    # Apply filters
    filtered_data = data.copy()
    
    if st.session_state.selected_airline != "All":
        filtered_data = filtered_data[filtered_data['airline'] == st.session_state.selected_airline]
    
    if len(st.session_state.date_range) == 2:
        start_date, end_date = st.session_state.date_range
        filtered_data = filtered_data[
            (filtered_data['tweet_created'].dt.date >= start_date) & 
            (filtered_data['tweet_created'].dt.date <= end_date)
        ]
    
    if st.session_state.sentiment_filter:
        filtered_data = filtered_data[filtered_data['airline_sentiment'].isin(st.session_state.sentiment_filter)]
    
    # ===== DASHBOARD HEADER =====
    st.title("✈️ Airline Sentiment Analysis")
    st.markdown(f"""
    <div class='metric-card'>
        Analyzing <strong>{len(filtered_data):,}</strong> tweets | Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M')}
    </div>
    """, unsafe_allow_html=True)
    
    # ===== KEY METRICS =====
    col1, col2, col3 = st.columns(3)
    with col1:
        pos_count = len(filtered_data[filtered_data['airline_sentiment'] == 'positive'])
        st.metric("Positive Tweets", pos_count, f"{pos_count/len(filtered_data):.1%}")
    with col2:
        neu_count = len(filtered_data[filtered_data['airline_sentiment'] == 'neutral'])
        st.metric("Neutral Tweets", neu_count, f"{neu_count/len(filtered_data):.1%}")
    with col3:
        neg_count = len(filtered_data[filtered_data['airline_sentiment'] == 'negative'])
        st.metric("Negative Tweets", neg_count, f"{neg_count/len(filtered_data):.1%}")
    
    # ===== VISUALIZATIONS =====
    tab1, tab2, tab3, tab4 = st.tabs(["📈 Trends", "🛫 Airlines", "☁️ Word Clouds", "🔍 Live Analysis"])
    
    with tab1:
        st.subheader("Sentiment Over Time")
        time_df = filtered_data.set_index('tweet_created').resample('D')['airline_sentiment'].value_counts().unstack()
        fig = px.area(
            time_df,
            color_discrete_map={
                'positive': '#10b981',
                'neutral': '#f59e0b',
                'negative': '#ef4444'
            }
        )
        fig.update_layout(
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font={'color': 'white'}
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with tab2:
        st.subheader("Airline Performance")
        airline_df = filtered_data.groupby(['airline', 'airline_sentiment']).size().unstack()
        fig = px.bar(
            airline_df,
            barmode='group',
            color_discrete_map={
                'positive': '#10b981',
                'neutral': '#f59e0b',
                'negative': '#ef4444'
            }
        )
        fig.update_layout(
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font={'color': 'white'}
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with tab3:
        st.subheader("Word Cloud Analysis")
        sentiment = st.radio(
            "Select sentiment:",
            ["positive", "neutral", "negative"],
            horizontal=True
        )
        
        sentiment_data = filtered_data[filtered_data['airline_sentiment'] == sentiment]
        if not sentiment_data.empty:
            text_data = ' '.join(sentiment_data['text'])
            fig = generate_wordcloud(text_data, sentiment)
            st.pyplot(fig)
            plt.close(fig)
        else:
            st.warning(f"No {sentiment} sentiment data available.")
    
    with tab4:
        st.subheader("🔍 Live Tweet Sentiment Analysis")
        user_tweet = st.text_area(
            "Enter a tweet to analyze:",
            "The flight was great and the service was excellent!",
            height=100
        )
        
        if st.button("Analyze Sentiment", type="primary"):
            with st.spinner("Analyzing..."):
                sentiment, score = analyze_sentiment(user_tweet)
                
                # Display sentiment with colored box
                sentiment_class = sentiment.lower()
                st.markdown(
                    f"""
                    <div class="sentiment-box {sentiment_class}">
                        <h3>Sentiment: <strong>{sentiment.capitalize()}</strong></h3>
                        <p>Polarity Score: <strong>{score:.2f}</strong></p>
                        <p>{"😊" if sentiment == "positive" else "😐" if sentiment == "neutral" else "😠"}</p>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

# ==================== MAIN APP ====================
def main():
    apply_dark_theme()
    render_sidebar()
    render_main_content()

if __name__ == "__main__":
    main()