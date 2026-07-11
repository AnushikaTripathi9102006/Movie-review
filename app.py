import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch.nn.functional as F

st.set_page_config(
    page_title="Movie Review Sentiment Analysis",
    page_icon="🎬",
    layout="centered"
)

@st.cache_resource
def load_model():
    tokenizer = AutoTokenizer.from_pretrained("model")
    model = AutoModelForSequenceClassification.from_pretrained("model")
    return tokenizer, model

tokenizer, model = load_model()

st.title("🎬 Netflix Movie Review Sentiment Analyzer")
st.write("Analyze the sentiment of a movie review using a fine-tuned Large Language Model.")

review = st.text_area(
    "Enter Movie Review",
    height=220,
    placeholder="Type your review here..."
)

if st.button("Predict Sentiment"):

    if review.strip() == "":
        st.warning("Please enter a review.")
        st.stop()

    inputs = tokenizer(
        review,
        truncation=True,
        padding=True,
        max_length=512,
        return_tensors="pt"
    )

    with torch.no_grad():
        outputs = model(**inputs)

    probs = F.softmax(outputs.logits, dim=1)

    prediction = torch.argmax(probs, dim=1).item()

    confidence = probs[0][prediction].item()

    labels = {
        0: "NEGATIVE",
        1: "POSITIVE"
    }

    sentiment = labels[prediction]

    if sentiment == "POSITIVE":
        st.success(f"😊 Sentiment : {sentiment}")
    else:
        st.error(f"😞 Sentiment : {sentiment}")

    st.metric(
        "Confidence",
        f"{confidence*100:.2f}%"
    )

    st.progress(float(confidence))

    st.subheader("Review")

    st.write(review)
