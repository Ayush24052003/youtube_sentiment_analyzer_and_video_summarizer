import streamlit as st
import plotly.express as px

from scripts.youtube_sentiment_analyzer import (
    load_model_components,
    get_youtube_comments,
    predict_youtube_sentiments,
    generate_word_cloud,
    get_transcript_summary
)

# Streamlit Config
st.set_page_config(
    page_title="YouTube Comment Sentiment Analyzer & Summarizer",
    page_icon="▶️",
    layout="wide"
)

st.title("▶️ YouTube Comment Sentiment Analyzer & Summarizer")
st.markdown("Analyze comment sentiments and summarize a video's transcript using AI.")

# Loading Model Components
@st.cache_resource
def get_loaded_model_components():
    return load_model_components(
        model_path="../models/best_bilstm_model.h5",
        tokenizer_path="../models/tokenizer.pkl",
        label_encoder_path="../models/label_encoder.pkl"
    )

model, tokenizer, label_encoder = get_loaded_model_components()
if not all([model, tokenizer, label_encoder]):
    st.stop()

# Load API Keys
try:
    youtube_api_key = st.secrets["youtube_api_key"]
    together_api_key = st.secrets["together_api_key"]
except KeyError as e:
    st.error(f"{str(e)} key not found in Streamlit secrets.")
    st.markdown("Ensure `.streamlit/secrets.toml` contains both keys.")
    st.stop()

# User Input
video_url = st.text_input("Enter YouTube Video URL:")

col1, col2 = st.columns(2)

# Comment Sentiment Analysis
with col1:
    max_comments = st.slider("Max comments to fetch:", 50, 1000, 200, step=50)
    if st.button("Analyze Comments", use_container_width=True):
        if not video_url:
            st.warning("Please enter a YouTube video URL.")
        else:
            with st.spinner(f"Fetching up to {max_comments} comments..."):
                comments = get_youtube_comments(video_url, youtube_api_key, max_comments=max_comments)
                if not comments:
                    st.info("No comments found or an error occurred.")
                    st.session_state.pop("last_comments_for_wordcloud", None)
                else:
                    result_df = predict_youtube_sentiments(comments, model, tokenizer, label_encoder)
                    st.success("Analysis Complete!")

                    st.session_state["last_comments_for_wordcloud"] = comments

                    # Sentiment Distribution Chart
                    st.subheader("Sentiment Distribution")
                    sentiment_counts = result_df["Predicted Sentiment"].value_counts(normalize=True) * 100
                    sentiment_counts = sentiment_counts.reindex(["positive", "negative", "neutral"], fill_value=0)

                    fig = px.bar(
                        x=sentiment_counts.index,
                        y=sentiment_counts.values,
                        color=sentiment_counts.index,
                        color_discrete_map={"positive": "green", "negative": "red", "neutral": "blue"},
                        labels={"x": "Sentiment", "y": "Percentage (%)"},
                        title="Sentiment Distribution"
                    )
                    fig.update_traces(texttemplate='%{y:.2f}%', textposition='outside')
                    st.plotly_chart(fig, use_container_width=True)

                    # Data Table
                    st.subheader("Individual Comment Sentiments")
                    st.dataframe(result_df, use_container_width=True)

                    st.download_button(
                        "Download CSV",
                        data=result_df.to_csv(index=False).encode("utf-8"),
                        file_name="youtube_sentiment_analysis.csv",
                        mime="text/csv"
                    )

# Transcript Summary Section
with col2:
    max_summary_words = st.slider("Summary length (words):", 50, 300, 150, step=10)
    if st.button("Summarize Video Transcript", use_container_width=True):
        if not video_url:
            st.warning("Please enter a YouTube video URL.")
        else:
            with st.spinner("Fetching transcript and summarizing..."):
                summary, transcript_text = get_transcript_summary(
                    video_url,
                    together_api_key,
                    max_summary_words=max_summary_words
                )

                if summary and not summary.startswith("❌") and "error" not in summary.lower():
                    st.subheader("Transcript Summary")
                    st.write(summary)

                    with st.expander("📄 View Full Transcript"):
                        st.text_area("Transcript Text", transcript_text, height=300)
                else:
                    st.warning(summary or "Transcript could not be fetched or summarized.")

# Word Cloud Generation
st.markdown("---")
st.subheader("Generate Word Cloud")
if "last_comments_for_wordcloud" in st.session_state:
    if st.button("Generate Word Cloud"):
        with st.spinner("Generating word cloud..."):
            image = generate_word_cloud(st.session_state["last_comments_for_wordcloud"])
            if image:
                st.image(image, caption="Top Frequent Words", use_column_width=True)
            else:
                st.info("Word cloud could not be generated. Try analyzing more comments.")
else:
    st.info("Run sentiment analysis first to enable word cloud generation.")
