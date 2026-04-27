# # #This Streamlit app allows users to:
# # #--predict sentiment from text input
# # #--Upload CSV files and analyze sentiment
# # #--Visualize result using charts and word cloud
# # import os
# # os.environ["Transformers_No_TF"]='1'


# # import streamlit as st
# # import pandas as pd
# # import re #Text Cleaning(remove list,symbol,etc)
# # import matplotlib.pyplot as plt
# # from wordcloud import WordCloud # Used to create word cloud(show most frequent word)
# # from transformers import pipeline 


# # st.set_page_config(
# #     page_title="AI Sentiment Analyzer"
# #     page_icon"⚝"
# #     layout="wide"
# # )

# # st.title("")

# # =========================================================
# # AI SENTIMENT ANALYSIS WEB APP (PROFESSIONAL VERSION)
# # =========================================================
# # This Streamlit app allows users to:
# # - Predict sentiment from text input
# # - Upload CSV files and analyze sentiment
# # - Visualize results using charts and word cloud
# # =========================================================

# # ==============================
# # ⚙️ 1. ENVIRONMENT SETUP
# # ==============================

# import os
# os.environ["TRANSFORMERS_NO_TF"] = "1"   # Disable TensorFlow (use PyTorch only)

# # 2. IMPORT LIBRARIES

# import streamlit as st
# import pandas as pd
# import re
# from transformers import pipeline
# import matplotlib.pyplot as plt
# from wordcloud import WordCloud

# # 3. PAGE CONFIGURATION
# st.set_page_config(
#     page_title="AI Sentiment Analyzer",
#     page_icon="🧠",
#     layout="wide"
# )

# # 🏷️ 4. TITLE & DESCRIPTION

# st.title("🧠 AI Sentiment Analysis Dashboard")
# st.markdown("""
# Analyze text sentiment using Transformer-based models.  
# Supports:
# - ✍️ Live text prediction  
# - 📂 CSV file analysis  
# - 📊 Data visualization  
# """)


# # 🤗 5. LOAD AI MODEL (CACHED)

# @st.cache_resource
# def load_model():
#     """
#     Load Hugging Face sentiment analysis model
#     Uses DistilBERT (fast + efficient)
#     """
#     return pipeline(
#         "sentiment-analysis",
#         model="distilbert-base-uncased-finetuned-sst-2-english",
#         framework="pt"   # Force PyTorch backend
#     )

# # Load model once (cached)
# classifier = load_model()


# # ==============================
# # 🧹 6. TEXT CLEANING FUNCTION
# # ==============================

# def clean_text(text):
#     """
#     Clean input text by:
#     - Lowercasing
#     - Removing URLs, mentions, hashtags
#     - Removing special characters
#     """
#     text = str(text).lower()
#     text = re.sub(r"http\S+", "", text)
#     text = re.sub(r"@\w+", "", text)
#     text = re.sub(r"#\w+", "", text)
#     text = re.sub(r"[^a-zA-Z\s]", "", text)
#     return text.strip()


# # ==============================
# # 🔮 7. SENTIMENT PREDICTION
# # ==============================

# def predict_sentiment(text):
#     """
#     Predict sentiment for a single text input
#     """
#     result = classifier(text)[0]

#     if result['label'] == 'POSITIVE':
#         return 'positive'
#     elif result['label'] == 'NEGATIVE':
#         return 'negative'
#     else:
#         return 'neutral'   # fallback (model usually gives 2 classes)


# # ==============================
# # ✍️ 8. LIVE TEXT PREDICTION
# # ==============================

# st.subheader("✍️ Live Sentiment Prediction")

# user_input = st.text_area("Enter your text here:")

# if st.button("Predict Sentiment"):
#     if user_input.strip():
#         cleaned_text = clean_text(user_input)

#         with st.spinner("Analyzing sentiment..."):
#             prediction = predict_sentiment(cleaned_text)

#         st.success(f"Prediction: {prediction.upper()}")
#     else:
#         st.warning("⚠️ Please enter some text")


# # ==============================
# # 📂 9. CSV FILE ANALYSIS
# # ==============================

# st.subheader("📂 Upload CSV for Batch Analysis")

# uploaded_file = st.file_uploader(
#     "Upload CSV file (must contain 'text' column)",
#     type=["csv"]
# )

# if uploaded_file is not None:
#     try:
#         df = pd.read_csv(uploaded_file)

#         # Validate columns
#         if 'text' not in df.columns:
#             st.error("❌ CSV must contain a 'text' column")
#             st.stop()

#         # Optional: rename sentiment column if exists
#         if 'airline_sentiment' in df.columns:
#             df.rename(columns={'airline_sentiment': 'label'}, inplace=True)

#         st.write("### 📊 Data Preview")
#         st.dataframe(df.head())

#         # ==============================
#         # 🧹 CLEAN DATA
#         # ==============================
#         df['text'] = df['text'].apply(clean_text)

#         # ==============================
#         # ⚡ PERFORMANCE OPTIMIZATION
#         # ==============================
#         # Limit rows for fast processing
#         df_sample = df.head(200).copy()

#         # Batch prediction (faster than apply)
#         texts = df_sample['text'].tolist()

#         with st.spinner("Processing data..."):
#             results = classifier(texts, batch_size=16)

#         # Convert predictions
#         df_sample['predicted'] = [
#             'positive' if r['label'] == 'POSITIVE' else 'negative'
#             for r in results
#         ]

#         st.success("✅ Prediction Completed")

#         # ==============================
#         # 📈 SENTIMENT DISTRIBUTION
#         # ==============================
#         st.subheader("📈 Sentiment Distribution")

#         fig1, ax1 = plt.subplots()
#         df_sample['predicted'].value_counts().plot.pie(
#             autopct='%1.1f%%',
#             ax=ax1
#         )
#         ax1.set_ylabel("")
#         st.pyplot(fig1)

#         # ==============================
#         # ☁️ WORD CLOUD
#         # ==============================
#         st.subheader("☁️ Word Cloud")

#         text_data = " ".join(df_sample['text'])

#         wordcloud = WordCloud(
#             width=800,
#             height=400,
#             background_color='white'
#         ).generate(text_data)

#         fig2, ax2 = plt.subplots()
#         ax2.imshow(wordcloud)
#         ax2.axis("off")

#         st.pyplot(fig2)

#     except Exception as e:
#         st.error(f"❌ Error processing file: {e}")


# # ==============================
# # 📌 10. FOOTER
# # ==============================

# st.markdown("---")
# st.markdown("""
# 🚀 Built using **Hugging Face Transformers** and **Streamlit**  
# ⚡ Model: DistilBERT (Fast & Efficient NLP Model)
# """)






# # ==============================
# # ⚙️ 1. ENV SETUP
# # ==============================
# import os
# os.environ["TRANSFORMERS_NO_TF"] = "1"

# # ==============================
# # 📦 2. IMPORTS
# # ==============================
# import streamlit as st
# import pandas as pd
# import re
# from transformers import pipeline
# import matplotlib.pyplot as plt
# from wordcloud import WordCloud

# # ==============================
# # 🎨 3. PAGE CONFIG + CUSTOM CSS
# # ==============================
# st.set_page_config(
#     page_title="SentimentAI",
#     page_icon="🚀",
#     layout="wide"
# )

# # Custom Styling (Landing Page Feel)
# st.markdown("""
# <style>
# .main {
#     background: linear-gradient(135deg, #0f172a, #1e293b);
#     color: white;
# }
# h1, h2, h3 {
#     color: #38bdf8;
# }
# .stButton>button {
#     background: linear-gradient(90deg, #06b6d4, #3b82f6);
#     color: white;
#     border-radius: 12px;
#     height: 3em;
#     width: 100%;
# }
# .card {
#     padding: 20px;
#     border-radius: 15px;
#     background: #111827;
#     box-shadow: 0 4px 20px rgba(0,0,0,0.3);
# }
# </style>
# """, unsafe_allow_html=True)

# # ==============================
# # 🏆 4. HERO SECTION
# # ==============================
# st.markdown("""
# # 🚀 SentimentAI
# ### Understand emotions in text instantly with AI
# """)

# col1, col2, col3 = st.columns(3)
# col1.metric("⚡ Speed", "Fast", "+Batch Processing")
# col2.metric("🧠 Model", "DistilBERT", "Transformer")
# col3.metric("📊 Accuracy", "~90%", "SST-2")

# st.markdown("---")

# # ==============================
# # 🤗 5. LOAD MODEL
# # ==============================
# @st.cache_resource
# def load_model():
#     return pipeline(
#         "sentiment-analysis",
#         model="distilbert-base-uncased-finetuned-sst-2-english",
#         framework="pt"
#     )

# classifier = load_model()

# # ==============================
# # 🧹 6. CLEAN TEXT
# # ==============================
# def clean_text(text):
#     text = str(text).lower()
#     text = re.sub(r"http\S+|@\w+|#\w+", "", text)
#     text = re.sub(r"[^a-zA-Z\s]", "", text)
#     return text.strip()

# # ==============================
# # 🔮 7. PREDICT
# # ==============================
# def predict(text):
#     result = classifier(text)[0]
#     label = result['label']
#     score = round(result['score'] * 100, 2)

#     sentiment = "positive" if label == "POSITIVE" else "negative"
#     return sentiment, score

# # ==============================
# # 🧭 8. NAVIGATION TABS
# # ==============================
# tab1, tab2 = st.tabs(["✍️ Live Analysis", "📂 Batch Analysis"])

# # ==============================
# # ✍️ TAB 1: LIVE
# # ==============================
# with tab1:
#     st.markdown("## ✍️ Real-Time Sentiment Detection")

#     user_input = st.text_area("Enter text")

#     if st.button("Analyze Sentiment"):
#         if user_input.strip():
#             cleaned = clean_text(user_input)

#             with st.spinner("Analyzing..."):
#                 sentiment, score = predict(cleaned)

#             st.markdown(f"""
#             <div class="card">
#             <h3>Result: {sentiment.upper()}</h3>
#             <p>Confidence: {score}%</p>
#             </div>
#             """, unsafe_allow_html=True)

#         else:
#             st.warning("Enter some text first.")

# # ==============================
# # 📂 TAB 2: CSV ANALYSIS
# # ==============================
# with tab2:
#     st.markdown("## 📊 Batch Sentiment Analysis")

#     uploaded = st.file_uploader("Upload CSV (must have 'text')", type=["csv"])

#     if uploaded:
#         df = pd.read_csv(uploaded)

#         if "text" not in df.columns:
#             st.error("CSV must contain 'text' column")
#             st.stop()

#         st.dataframe(df.head())

#         df['text'] = df['text'].apply(clean_text)

#         sample_size = st.slider("Select sample size", 50, 500, 200)
#         df_sample = df.head(sample_size)

#         if st.button("Run Analysis"):
#             with st.spinner("Processing..."):
#                 results = classifier(df_sample['text'].tolist(), batch_size=16)

#             df_sample['sentiment'] = [
#                 "positive" if r['label'] == "POSITIVE" else "negative"
#                 for r in results
#             ]

#             df_sample['confidence'] = [round(r['score']*100,2) for r in results]

#             # ==============================
#             # 📊 KPIs
#             # ==============================
#             pos = (df_sample['sentiment'] == 'positive').sum()
#             neg = (df_sample['sentiment'] == 'negative').sum()

#             c1, c2, c3 = st.columns(3)
#             c1.metric("Total", len(df_sample))
#             c2.metric("Positive", pos)
#             c3.metric("Negative", neg)

#             # ==============================
#             # 📈 PIE CHART
#             # ==============================
#             fig, ax = plt.subplots()
#             df_sample['sentiment'].value_counts().plot.pie(autopct='%1.1f%%', ax=ax)
#             ax.set_ylabel("")
#             st.pyplot(fig)

#             # ==============================
#             # ☁️ WORD CLOUD
#             # ==============================
#             text_data = " ".join(df_sample['text'])

#             wc = WordCloud(width=800, height=400, background_color='black').generate(text_data)

#             fig2, ax2 = plt.subplots()
#             ax2.imshow(wc)
#             ax2.axis("off")
#             st.pyplot(fig2)

#             # ==============================
#             # 📥 DOWNLOAD
#             # ==============================
#             csv = df_sample.to_csv(index=False).encode('utf-8')
#             st.download_button("⬇️ Download Results", csv, "results.csv")

# # ==============================
# # 📌 FOOTER
# # ==============================
# st.markdown("---")
# st.markdown("""
# <center>
# 🚀 Built with Transformers + Streamlit  
# 💡 Designed like a modern AI SaaS landing page  
# </center>
# """, unsafe_allow_html=True)








# ==============================
# ⚙️ ENV SETUP
# ==============================
import os
os.environ["TRANSFORMERS_NO_TF"] = "1"

# ==============================
# 📦 IMPORTS
# ==============================
import streamlit as st
import pandas as pd
import re
from transformers import pipeline
import matplotlib.pyplot as plt
from wordcloud import WordCloud

# ==============================
# 🎨 PAGE CONFIG + STYLING
# ==============================
st.set_page_config(
    page_title="SentimentAI",
    page_icon="◦",
    layout="wide"
)


st.markdown("""
<style>
body {
    background-color: #0b0f19;
}
.main {
    background: linear-gradient(145deg, #0b0f19, #111827);
    color: #e5e7eb;
}

h1 {
    font-weight: 600;
    letter-spacing: 0.5px;
}

h2, h3 {
    color: #9ca3af;
    font-weight: 500;
}

.stButton>button {
    background: #1f2937;
    color: #e5e7eb;
    border-radius: 10px;
    border: 1px solid #374151;
    height: 2.8em;
}
.stButton>button:hover {
    border: 1px solid #60a5fa;
}

.card {
    padding: 22px;
    border-radius: 14px;
    background: #111827;
    border: 1px solid #1f2937;
}

.metric {
    font-size: 14px;
    color: #9ca3af;
}

.divider {
    margin-top: 20px;
    margin-bottom: 20px;
    border-bottom: 1px solid #1f2937;
}

</style>
""", unsafe_allow_html=True)

# ==============================
# ✦ HERO SECTION
# ==============================
st.markdown("""
# 🛦ExpAIr 
#### 🌐Transformer-based sentiment intelligence
""")

col1, col2, col3 = st.columns(3)
col1.metric("Model", "DistilBERT")
col2.metric("Latency", "Low")
col3.metric("Mode", "Batch + Realtime")

st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

# ==============================
# 🤗 MODEL LOAD
# ==============================
@st.cache_resource
def load_model():
    return pipeline(
        "sentiment-analysis",
        model="distilbert-base-uncased-finetuned-sst-2-english",
        framework="pt"
    )

classifier = load_model()

# ==============================
# 🧹 CLEAN TEXT
# ==============================
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"http\S+|@\w+|#\w+", "", text)
    text = re.sub(r"[^a-zA-Z\s]", "", text)
    return text.strip()

# ==============================
# 🔮 PREDICTION
# ==============================
def predict(text):
    result = classifier(text)[0]
    label = result['label']
    score = round(result['score'] * 100, 2)

    sentiment = "positive" if label == "POSITIVE" else "negative"
    return sentiment, score

# ==============================
# NAVIGATION
# ==============================
tab1, tab2 = st.tabs(["Live Analysis", "Batch Processing"])

# ==============================
# ✦ LIVE ANALYSIS
# ==============================
with tab1:
    st.markdown("### ▸ Real-time Sentiment")

    user_input = st.text_area("Input text")

    if st.button("Run Analysis"):
        if user_input.strip():
            cleaned = clean_text(user_input)

            with st.spinner("Processing..."):
                sentiment, score = predict(cleaned)

            st.markdown(f"""
            <div class="card">
            <h3>{sentiment.upper()}</h3>
            <p class="metric">Confidence — {score}%</p>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.warning("Input required.")

# ==============================
# ✦ BATCH ANALYSIS
# ==============================
with tab2:
    st.markdown("### ▸ Dataset Analysis")

    uploaded = st.file_uploader("Upload CSV (requires 'text' column)", type=["csv"])

    if uploaded:
        df = pd.read_csv(uploaded)

        if "text" not in df.columns:
            st.error("Missing 'text' column.")
            st.stop()

        st.dataframe(df.head())

        df['text'] = df['text'].apply(clean_text)

        sample_size = st.slider("Sample size", 50, 500, 200)
        df_sample = df.head(sample_size)

        if st.button("Process Dataset"):
            with st.spinner("Running inference..."):
                results = classifier(df_sample['text'].tolist(), batch_size=16)

            df_sample['sentiment'] = [
                "positive" if r['label'] == "POSITIVE" else "negative"
                for r in results
            ]
            df_sample['confidence'] = [
                round(r['score'] * 100, 2) for r in results
            ]

            # ==============================
            # KPIs
            # ==============================
            pos = (df_sample['sentiment'] == 'positive').sum()
            neg = (df_sample['sentiment'] == 'negative').sum()

            c1, c2, c3 = st.columns(3)
            c1.metric("Total", len(df_sample))
            c2.metric("Positive", pos)
            c3.metric("Negative", neg)

            # ==============================
            # PIE
            # ==============================
            fig, ax = plt.subplots()
            df_sample['sentiment'].value_counts().plot.pie(
                autopct='%1.1f%%', ax=ax
            )
            ax.set_ylabel("")
            st.pyplot(fig)

            # ==============================
            # WORD CLOUD
            # ==============================
            text_data = " ".join(df_sample['text'])

            wc = WordCloud(
                width=800,
                height=400,
                background_color='#0b0f19'
            ).generate(text_data)

            fig2, ax2 = plt.subplots()
            ax2.imshow(wc)
            ax2.axis("off")
            st.pyplot(fig2)

            # ==============================
            # DOWNLOAD
            # ==============================
            csv = df_sample.to_csv(index=False).encode('utf-8')
            st.download_button("Download Results", csv, "results.csv")

# ==============================
# FOOTER
# ==============================
st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

st.markdown("""
<center style="color:#6b7280;">
⌁ Built with Transformer Models · Streamlit Interface  
</center>
""", unsafe_allow_html=True)
