#  YouTube Video Summarizer and Sentiment Analyzer

This project analyzes the sentiment of YouTube comments and generates a summary of the video transcript using Together.ai.

## Features

-  Fetches transcript using `yt-dlp` with Hindi-English fallback
-  Summarizes transcript using Together.ai
-  Analyzes YouTube comment sentiment using a BiLSTM + Attention model
-  Interactive UI using Streamlit
-  Visualizations including word cloud and comment breakdown

##  Project Structure
 
.streamlit/ – Streamlit configuration files

- secrets.toml – Stores API keys and other secrets

.venv/ – Python virtual environment

models – Contains trained models and preprocessing objects

- best_bilstm_model.h5 – BiLSTM model file

- label_encoder.pkl – Label encoder for sentiment classes

- tokenizer.pkl – Tokenizer used during training

notebook

- youtube_glove.ipynb – GloVe embedding integration

- ytb_sentiment.ipynb – Sentiment model development and training

scripts – Core Python scripts for the app

- streamlit_app.py – Streamlit frontend app

- youtube_sentiment_analyzer.py – Main analysis logic (loading model, predicting)

.gitignore – Lists files/folders to ignore in version control

README.md – Project overview and documentation

requirements.txt – List of required Python libraries
