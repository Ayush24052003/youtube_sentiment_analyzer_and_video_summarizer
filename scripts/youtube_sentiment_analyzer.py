import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Model
from tensorflow.keras import backend as K
import pickle
from googleapiclient.discovery import build  # For YouTube Data API
from googleapiclient.errors import HttpError  # Import HttpError to catch API specific errors
import re  # For cleaning comments fetched from YouTube
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import nltk
from nltk.corpus import stopwords
import string
import requests  # For making API calls to LLM (Together.ai)

MODEL_PATH = '../models/best_bilstm_model.h5'
TOKENIZER_PATH = '../models/tokenizer.pkl'
LABEL_ENCODER_PATH = '../models/label_encoder.pkl'

MAX_LEN_RNN_LSTM_GRU = 100  # Max length used for padding sequences
EMBEDDING_DIM_RNN_LSTM_GRU = 100  # Must match the GloVe dimension
VOCAB_SIZE_RNN_LSTM_GRU = 10000  # Max vocabulary size used in Tokenizer


# Custom Attention Layer
class Attention(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super(Attention, self).__init__(**kwargs)

    def build(self, input_shape):
        self.W = self.add_weight(name="att_weight", shape=(input_shape[-1], 1),
                                 initializer="normal", trainable=True)
        self.b = self.add_weight(name="att_bias", shape=(input_shape[1], 1),
                                 initializer="zeros", trainable=True)
        super(Attention, self).build(input_shape)

    def call(self, x):
        # x is (batch_size, sequence_length, features)
        e = K.tanh(K.dot(x, self.W) + self.b)  # (batch_size, sequence_length, 1)
        e = K.squeeze(e, axis=-1)  # (batch_size, sequence_length)
        alpha = K.softmax(e)  # (batch_size, sequence_length)
        alpha = K.expand_dims(alpha, axis=-1)  # (batch_size, sequence_length, 1)
        output = x * alpha  # (batch_size, sequence_length, features)
        output = K.sum(output, axis=1)  # (batch_size, features)
        return output

    def get_config(self):
        return super(Attention, self).get_config()


# Model Definition
def create_deep_bilstm_model(learning_rate=0.001, dropout_rate=0.3, lstm_units_1=64, lstm_units_2=32):

    lstm_input = Input(shape=(MAX_LEN_RNN_LSTM_GRU,))

    lstm_embedding = tf.keras.layers.Embedding(
        VOCAB_SIZE_RNN_LSTM_GRU,
        EMBEDDING_DIM_RNN_LSTM_GRU,
        input_length=MAX_LEN_RNN_LSTM_GRU,
    )(lstm_input)

    lstm_layer_1 = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(lstm_units_1, return_sequences=True))(
        lstm_embedding)
    lstm_dropout_1 = tf.keras.layers.Dropout(dropout_rate)(lstm_layer_1)
    lstm_layer_2 = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(lstm_units_2, return_sequences=True))(
        lstm_dropout_1)
    lstm_dropout_2 = tf.keras.layers.Dropout(dropout_rate)(lstm_layer_2)

    attention_output = Attention()(lstm_dropout_2)
    lstm_output = tf.keras.layers.Dense(3, activation='softmax')(attention_output)  # 3 classes for sentiment

    model = tf.keras.models.Model(inputs=lstm_input, outputs=lstm_output)
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model


# Loading Trained Model Components
def load_model_components(model_path=MODEL_PATH, tokenizer_path=TOKENIZER_PATH, label_encoder_path=LABEL_ENCODER_PATH):
    try:
        model = tf.keras.models.load_model(
            model_path,
            custom_objects={'Attention': Attention, 'create_deep_bilstm_model': create_deep_bilstm_model}
        )
        print(f"Model loaded from {model_path}")

        # Load Tokenizer
        with open(tokenizer_path, 'rb') as f:
            tokenizer = pickle.load(f)
        print(f"Tokenizer loaded from {tokenizer_path}")

        # Load LabelEncoder
        with open(label_encoder_path, 'rb') as f:
            label_encoder = pickle.load(f)
        print(f"LabelEncoder loaded from {label_encoder_path}")

        return model, tokenizer, label_encoder
    except FileNotFoundError as e:
        print(f"Error loading components: {e}. Make sure the files exist in the correct directory.")
        return None, None, None
    except Exception as e:
        print(f"An unexpected error occurred during loading: {e}")
        return None, None, None


# YouTube Comments Fetcher
def get_youtube_comments(video_url, api_key, max_comments=500):
    youtube = build('youtube', 'v3', developerKey=api_key)

    video_id = None
    patterns = [
        r'(?<=v=)[^&]+',
        r'(?<=youtu.be/)[^?]+',
        r'(?<=embed/)[^?]+',
        r'(?<=/v/)[^?]+'
    ]

    for pattern in patterns:
        video_id_match = re.search(pattern, video_url)
        if video_id_match:
            video_id = video_id_match.group(0)
            break

    if not video_id:
        print(f"Error: Could not extract video ID from the URL: {video_url}")
        return []

    comments = []
    next_page_token = None
    total_fetched = 0

    print(f"Attempting to fetch comments for video ID: {video_id} from URL: {video_url}")

    try:
        while True:
            request = youtube.commentThreads().list(
                part='snippet',
                videoId=video_id,
                maxResults=min(100, max_comments - total_fetched),
                pageToken=next_page_token
            )
            response = request.execute()

            if 'items' not in response or not response['items']:
                print("No more comment threads found for this video or no comments at all.")
                break

            for item in response['items']:
                if 'topLevelComment' in item['snippet'] and 'snippet' in item['snippet']['topLevelComment']:
                    comment_text = item['snippet']['topLevelComment']['snippet']['textDisplay']
                    comments.append(comment_text)
                    total_fetched += 1
                    if total_fetched >= max_comments:
                        break
                else:
                    print("Skipping a comment item due to unexpected structure.")

            next_page_token = response.get('nextPageToken')
            if not next_page_token or total_fetched >= max_comments:
                break
    except HttpError as e:
        error_content = e.content.decode('utf-8')
        print(f"YouTube API Error (HTTP {e.resp.status}): {error_content}")
        if "commentsDisabled" in error_content:
            print("Reason: Comments are likely disabled for this video.")
        elif "videoNotFound" in error_content:
            print("Reason: Video not found or is private/deleted.")
        elif "quotaExceeded" in error_content:
            print("Reason: API quota exceeded. Try again later or check your Google Cloud Console.")
        elif "keyInvalid" in error_content or "accessNotConfigured" in error_content:
            print("Reason: API Key is invalid or YouTube Data API v3 is not enabled for your project.")
        else:
            print("An unknown API error occurred.")
        return []
    except Exception as e:
        print(f"An unexpected error occurred during comment fetching: {e}")
        return []

    print(f"Successfully fetched {len(comments)} comments.")
    return comments


# Prediction Function for Dashboard
def predict_youtube_sentiments(comments_list, model, tokenizer, label_encoder):

    if not comments_list:
        return pd.DataFrame(columns=['Comment', 'Predicted Sentiment'])

    comments_list = [str(c).fillna('') if pd.isna(c) else str(c) for c in comments_list]

    sequences = tokenizer.texts_to_sequences(comments_list)
    padded_sequences = pad_sequences(sequences, maxlen=MAX_LEN_RNN_LSTM_GRU, padding='post', truncating='post')

    predictions = model.predict(padded_sequences)
    predicted_classes = np.argmax(predictions, axis=-1)

    predicted_sentiments = label_encoder.inverse_transform(predicted_classes)

    results_df = pd.DataFrame({
        'Comment': comments_list,
        'Predicted Sentiment': predicted_sentiments
    })
    return results_df


# Word Cloud Generation Function
def generate_word_cloud(comments_list):

    if not comments_list:
        return None

    text = " ".join(comments_list)

    text = text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    stop_words = set(stopwords.words('english'))
    words = text.split()
    filtered_words = [word for word in words if word not in stop_words and word.isalpha()]
    cleaned_text = " ".join(filtered_words)

    if not cleaned_text.strip():
        return None

    wordcloud = WordCloud(
        width=800,
        height=400,
        background_color='white',
        colormap='viridis',
        min_font_size=10
    ).generate(cleaned_text)

    return wordcloud.to_image()


# Get Video Transcript Function
import subprocess
import json
import requests
import glob
import re
from langdetect import detect

PREFERRED_LANGS = ["en", "hi"]

def get_video_id(video_url):
    patterns = [
        r"(?:v=|\/)([0-9A-Za-z_-]{11})",
        r"youtu\.be\/([0-9A-Za-z_-]{11})",
        r"youtube\.com\/embed\/([0-9A-Za-z_-]{11})",
        r"youtube\.com\/live\/([0-9A-Za-z_-]{11})"
    ]
    for pattern in patterns:
        match = re.search(pattern, video_url)
        if match:
            return match.group(1)
    return None

def get_transcript_text(video_url: str, lang: str) -> str:
    print(f"🌐 Trying to fetch transcript in language: {lang}")
    command = [
        "yt-dlp",
        "--write-auto-sub",
        "--sub-lang", lang,
        "--sub-format", "json3",
        "--skip-download",
        "--output", "%(id)s.%(ext)s",
        video_url
    ]
    result = subprocess.run(command, capture_output=True, text=True)
    if result.returncode != 0:
        raise Exception("yt-dlp failed.")

    video_id = get_video_id(video_url)
    if not video_id:
        raise Exception("Could not extract video ID.")
    json_files = glob.glob(f"{video_id}*.json3")
    if not json_files:
        raise Exception(f" Transcript not found in {lang}. Trying next language...")

    with open(json_files[0], "r", encoding="utf-8") as f:
        data = json.load(f)

    transcript = []
    for event in data.get("events", []):
        if "segs" in event:
            for seg in event["segs"]:
                text = seg.get("utf8", "").strip()
                if text:
                    transcript.append(text)
    return " ".join(transcript)

def clean_and_detect_language(text: str):
    words = [w.strip() for w in re.findall(r'\b\w+\b', text) if len(w.strip()) > 1]
    cleaned = " ".join(words)
    try:
        lang = detect(cleaned)
    except:
        lang = "unknown"
    return cleaned.strip(), lang

def summarize_text_with_llm(transcript_text, together_api_key, max_words=150):
    prompt = (
        "You are a multilingual assistant. Given the following YouTube transcript (in either Hindi or English), "
        f"summarize it clearly in English in about {max_words} words, keeping all important details:\n\n"
        f"{transcript_text}\n\nSummary:"
    )

    headers = {
        "Authorization": f"Bearer {together_api_key}",
        "Content-Type": "application/json"
    }

    payload = {
        "model": "deepseek-ai/DeepSeek-R1-0528",
        "prompt": prompt,
        "max_tokens": int(max_words * 1.5),
        "temperature": 0.5,
        "top_p": 0.9,
        "stop": ["</s>"]
    }

    try:
        print("Sending request to Together API...")
        response = requests.post("https://api.together.xyz/v1/completions", headers=headers, json=payload)
        print("Status code:", response.status_code)
        print("🧾 Response text:", response.text[:1000])  # Print first 1000 chars
    except Exception as e:
        print("Request to Together AI failed:", str(e))
        return f"Request failed: {str(e)}"

    if response.status_code == 200:
        try:
            return response.json()["choices"][0]["text"].strip()
        except Exception as parse_error:
            print("Failed to parse response:", str(parse_error))
            return "Failed to parse response from Together AI."
    else:
        return f"Together API error: {response.status_code} - {response.text}"


def get_transcript_summary(video_url, together_api_key, max_summary_words=150):
    transcript_text = None
    for lang in PREFERRED_LANGS:
        try:
            transcript_text = get_transcript_text(video_url, lang)
            print(f"Transcript fetched in language: {lang}")
            break
        except Exception as e:
            print(str(e))

    if not transcript_text:
        return None, "Could not fetch transcript in any preferred language."

    cleaned_transcript, detected_lang = clean_and_detect_language(transcript_text)
    print(f"Detected transcript language: {detected_lang}")

    word_count = len(cleaned_transcript.split())
    print(f" Word count of cleaned transcript: {word_count}")
    if word_count < 30:
        return None, "Transcript too short or corrupted. Skipping summarization."

    print("Sending to Together AI for summarization...")
    summary = summarize_text_with_llm(cleaned_transcript, together_api_key, max_words=max_summary_words)
    return summary, cleaned_transcript

    # Clean up transcript files
    video_id = get_video_id(video_url)
    if video_id:
        for file in glob.glob(f"{video_id}*.json3"):
            try:
                os.remove(file)
            except Exception as e:
                print(f"Couldn't delete temporary file {file}: {e}")


