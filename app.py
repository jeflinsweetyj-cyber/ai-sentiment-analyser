import streamlit as st
import torch
from transformers import pipeline
import logging
import time

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Set page config
st.set_page_config(
    page_title="Sentiment Analyzer",
    page_icon="😊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Cache the model loading to avoid reloading on every interaction
@st.cache_resource
def load_sentiment_model():
    """Load the sentiment analysis model with error handling."""
    try:
        with st.spinner("Loading sentiment analysis model..."):
            # Use CPU if CUDA is not available
            device = 0 if torch.cuda.is_available() else -1
            
            pipe = pipeline(
                "text-classification",
                model="distilbert-base-uncased-finetuned-sst-2-english",
                device=device,
                return_all_scores=True
            )
            logger.info("Model loaded successfully")
            return pipe
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        logger.error(f"Model loading error: {str(e)}")
        return None

def analyze_sentiment(text, model):
    """Analyze sentiment with error handling."""
    try:
        if not text or not text.strip():
            return None, "Please enter some text to analyze."
        
        # Limit text length to avoid memory issues
        if len(text) > 5000:
            text = text[:5000]
            st.warning("Text truncated to 5000 characters for processing.")
        
        # Get predictions
        result = model(text)
        return result, None
        
    except Exception as e:
        error_msg = f"Error analyzing sentiment: {str(e)}"
        logger.error(error_msg)
        return None, error_msg

def display_results(results):
    """Display sentiment analysis results."""
    if not results:
        return
    
    # Get the results (returns list of predictions)
    predictions = results[0] if isinstance(results, list) else results
    
    # Create columns for better layout
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Sentiment Analysis Results")
        
        # Display results with colored metrics
        for pred in predictions:
            label = pred['label']
            score = pred['score']
            confidence = score * 100
            
            # Color coding based on sentiment
            if label == "POSITIVE":
                st.success(f"✅ **{label}**: {confidence:.2f}%")
            else:
                st.error(f"❌ **{label}**: {confidence:.2f}%")
    
    with col2:
        st.subheader("Confidence Visualization")
        
        # Create a simple bar chart
        labels = [pred['label'] for pred in predictions]
        scores = [pred['score'] for pred in predictions]
        
        chart_data = {
            'Sentiment': labels,
            'Confidence': scores
        }
        
        st.bar_chart(data=chart_data, x='Sentiment', y='Confidence')

def main():
    """Main application function."""
    st.title("🎭 Sentiment Analyzer")
    st.markdown("---")
    
    # Sidebar with information
    with st.sidebar:
        st.header("About")
        st.info(
            "This app uses DistilBERT model fine-tuned on Stanford Sentiment Treebank "
            "to classify text as positive or negative."
        )
        
        st.header("Features")
        st.markdown("""
        - Real-time sentiment analysis
        - Confidence scores for predictions
        - Support for long texts (up to 5000 chars)
        - Error handling and validation
        """)
        
        st.header("Model Info")
        st.code("distilbert-base-uncased-finetuned-sst-2-english")
    
    # Load model
    model = load_sentiment_model()
    
    if model is None:
        st.error("Failed to load the sentiment analysis model. Please refresh the page.")
        return
    
    # Main interface
    st.header("Enter Text for Analysis")
    
    # Text input options
    input_method = st.radio(
        "Choose input method:",
        ["Text Area", "File Upload"],
        horizontal=True
    )
    
    text_to_analyze = ""
    
    if input_method == "Text Area":
        text_to_analyze = st.text_area(
            "Enter your text here:",
            placeholder="Type or paste your text here...",
            height=150,
            help="Enter any text you want to analyze for sentiment."
        )
        
    else:
        uploaded_file = st.file_uploader(
            "Upload a text file",
            type=['txt'],
            help="Upload a .txt file to analyze its content."
        )
        
        if uploaded_file is not None:
            try:
                text_to_analyze = uploaded_file.read().decode('utf-8')
                st.text_area("File content:", text_to_analyze, height=150, disabled=True)
            except Exception as e:
                st.error(f"Error reading file: {str(e)}")
    
    # Analysis button and results
    if st.button("🔍 Analyze Sentiment", type="primary"):
        if text_to_analyze:
            with st.spinner("Analyzing sentiment..."):
                start_time = time.time()
                results, error = analyze_sentiment(text_to_analyze, model)
                end_time = time.time()
                
                if error:
                    st.error(error)
                else:
                    st.success(f"Analysis completed in {end_time - start_time:.2f} seconds")
                    display_results(results)
        else:
            st.warning("Please enter some text to analyze.")
    
    # Example texts
    st.header("Try These Examples")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("😊 Positive Example"):
            st.session_state['example_text'] = "I absolutely love this new feature! It's amazing and works perfectly."
    
    with col2:
        if st.button("😞 Negative Example"):
            st.session_state['example_text'] = "This is terrible and frustrating. I hate dealing with these bugs."
    
    with col3:
        if st.button("🤔 Mixed Example"):
            st.session_state['example_text'] = "The interface looks good but the performance could be better."
    
    # Display example if selected
    if 'example_text' in st.session_state:
        st.text_area("Example text:", st.session_state['example_text'], height=100, disabled=True)
        
        if st.button("Analyze Example"):
            with st.spinner("Analyzing example..."):
                results, error = analyze_sentiment(st.session_state['example_text'], model)
                
                if error:
                    st.error(error)
                else:
                    display_results(results)

# Additional error handling for the entire app
if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        st.error(f"An unexpected error occurred: {str(e)}")
        logger.error(f"Application error: {str(e)}")
        st.info("Please refresh the page and try again.")