import streamlit as st
import pandas as pd
import pickle
from io import StringIO
import matplotlib.pyplot as plt


# 1. Load your trained model and vectorizer
@st.cache_resource
def load_model_and_vectorizer():
    with open("model.pkl", "rb") as f:
        model = pickle.load(f)
    with open("vectorizer.pkl", "rb") as f:
        vectorizer = pickle.load(f)
    return model, vectorizer


# --- Streamlit UI Setup ---

# Page background styling
page_bg_img = """
<style>
.stApp {
    background-image: url("https://boast.io/wp-content/uploads/2022/05/12-Testimonial-Guidelines-to-Ensure-Youre-Not-Breaking-the-Law.jpg.webp");
    background-size: cover;
    background-position: center;
    background-repeat: no-repeat;
    background-attachment: fixed;
}
</style>
"""
st.markdown(page_bg_img, unsafe_allow_html=True)
model, vectorizer = load_model_and_vectorizer()

st.title("Review Sentiment Detection Dashboard")

st.sidebar.header("Input mode")
input_mode = st.sidebar.selectbox(
    "Choose input type",
    ["Upload CSV/Excel", "Single text review", "Voice to text (manual paste)"]
)


# Helper function to predict and create pie data
def get_sentiment_counts(text_series):
    # Convert to string in case of NaNs etc.
    text_series = text_series.astype(str)

    # Transform text using vectorizer
    X = vectorizer.transform(text_series)

    # Predict using model
    preds = model.predict(X)
    df_res = pd.DataFrame({"sentiment": preds})
    counts = df_res["sentiment"].value_counts().reset_index()
    counts.columns = ["sentiment", "count"]
    counts["percentage"] = counts["count"] / counts["count"].sum() * 100
    return df_res, counts


# 2. Mode 1: Upload CSV / Excel file
if input_mode == "Upload CSV/Excel":
    uploaded_file = st.file_uploader(
        "Upload review file (CSV or Excel)", type=["csv", "xlsx", "xls"]
    )

    text_column_name = st.text_input(
        "Name of the column containing review text",
        value="review"
    )

    if uploaded_file is not None:
        # Detect file type
        if uploaded_file.name.endswith(".csv"):
            # safer CSV read with encoding handling
            try:
                df = pd.read_csv(uploaded_file, encoding="utf-8")
            except UnicodeDecodeError:
                uploaded_file.seek(0)
                df = pd.read_csv(uploaded_file, encoding="latin-1")
        else:
            df = pd.read_excel(uploaded_file)

        st.subheader("Preview of uploaded data")
        st.dataframe(df.head())

        if text_column_name not in df.columns:
            st.error(f"Column '{text_column_name}' not found in file.")
        else:
            if st.button("Run Prediction"):
                review_texts = df[text_column_name]
                pred_df, pie_data = get_sentiment_counts(review_texts)

                # Attach predictions to original dataframe
                df["predicted_sentiment"] = pred_df["sentiment"]

                st.subheader("Data with predicted sentiment")
                st.dataframe(df.head())

                st.subheader("Sentiment distribution (pie chart)")
                # Matplotlib pie chart
                fig, ax = plt.subplots()
                ax.pie(
                    pie_data["count"],
                    labels=pie_data["sentiment"],
                    autopct="%1.1f%%",
                    startangle=90
                )
                ax.axis("equal")  # Equal aspect ratio ensures that pie is drawn as a circle.
                st.pyplot(fig)

                st.subheader("Counts and percentages")
                st.write(pie_data)

                # Download predictions
                csv_buffer = StringIO()
                df.to_csv(csv_buffer, index=False)
                st.download_button(
                    label="Download predictions as CSV",
                    data=csv_buffer.getvalue(),
                    file_name="predicted_reviews.csv",
                    mime="text/csv"
                )

# 3. Mode 2: Single text review
elif input_mode == "Single text review":
    user_text = st.text_area("Enter a single review text")
    if st.button("Predict sentiment"):
        if user_text.strip() == "":
            st.warning("Please enter some text.")
        else:
            _, counts = get_sentiment_counts(pd.Series([user_text]))
            pred_label = counts.loc[counts["count"].idxmax(), "sentiment"]
            st.success(f"Predicted sentiment: **{pred_label}**")

# 4. Mode 3: Voice to text (manual paste)
elif input_mode == "Voice to text (manual paste)":
    st.write("Use any speech-to-text tool to convert your voice message to text, then paste below.")
    voice_text = st.text_area("Paste transcribed text here")
    if st.button("Predict sentiment from voice text"):
        if voice_text.strip() == "":
            st.warning("Please paste some text.")
        else:
            _, counts = get_sentiment_counts(pd.Series([voice_text]))
            pred_label = counts.loc[counts["count"].idxmax(), "sentiment"]
            st.success(f"Predicted sentiment: **{pred_label}**")
