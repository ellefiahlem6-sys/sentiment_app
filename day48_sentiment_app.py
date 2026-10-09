import streamlit as st
from transformers import pipeline

# 1. Page Configuration
st.set_page_config(page_title="Sentiment Analyzer", page_icon="📊", layout="centered")

# 2. Load the Model (Using Cache to avoid reloading)
@st.cache_resource
def load_model():
    # We use a small, fast model for English sentiment analysis
    return pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")

st.title("📊 Customer Sentiment Analyzer")
st.markdown("Enter a customer review below, and the AI will analyze its sentiment instantly.")

# Load model with a spinner
with st.spinner("Loading AI Model... Please wait (this may take a minute on first run)..."):
    classifier = load_model()

# 3. User Input
review_text = st.text_area("📝 Customer Review (English):", height=150, placeholder="e.g., This product is amazing!")

# 4. Analyze Button
if st.button("🔍 Analyze Sentiment", type="primary"):
    if not review_text.strip():
        st.warning("⚠️ Please enter a review first.")
    else:
        with st.spinner("Analyzing..."):
            try:
                # Run the model
                result = classifier(review_text)[0]
                label = result['label']
                score = result['score'] * 100

                # Display Results
                st.markdown("---")
                st.subheader("Analysis Result")
                
                col1, col2 = st.columns(2)
                with col1:
                    if label == "POSITIVE":
                        st.success(f"😊 Sentiment: **{label}**")
                    else:
                        st.error(f"😞 Sentiment: **{label}**")
                
                with col2:
                    st.metric(label="Confidence", value=f"{score:.1f}%")
                    
            except Exception as e:
                st.error(f"❌ An error occurred: {e}")