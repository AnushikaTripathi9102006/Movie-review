import streamlit as st
from transformers import (
    pipeline,
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    AutoModelForQuestionAnswering
)
from transformers import pipeline as hf_pipeline

st.set_page_config(
    page_title="Netflix Review Analyzer",
    page_icon="🎬",
    layout="wide"
)

st.title("🎬 Netflix Movie Review Analyzer using LLMs")

st.write("Analyze Netflix movie reviews using Sentiment Analysis, Summarization and Question Answering.")

# -------------------------------
# Load Models
# -------------------------------

@st.cache_resource
def load_sentiment_model():
    return pipeline(
        "sentiment-analysis",
        model="distilbert-base-uncased-finetuned-sst-2-english"
    )

@st.cache_resource
def load_summary_model():
    tokenizer = AutoTokenizer.from_pretrained("cnicu/t5-small-booksum")
    model = AutoModelForSeq2SeqLM.from_pretrained("cnicu/t5-small-booksum")
    summarizer = hf_pipeline(
        "summarization",
        model=model,
        tokenizer=tokenizer
    )
    return summarizer

@st.cache_resource
def load_qa_model():
    tokenizer = AutoTokenizer.from_pretrained("deepset/minilm-uncased-squad2")
    model = AutoModelForQuestionAnswering.from_pretrained(
        "deepset/minilm-uncased-squad2"
    )
    qa = hf_pipeline(
        "question-answering",
        model=model,
        tokenizer=tokenizer
    )
    return qa


sentiment_model = load_sentiment_model()
summary_model = load_summary_model()
qa_model = load_qa_model()

# -----------------------------------
# Sidebar
# -----------------------------------

option = st.sidebar.selectbox(
    "Choose Task",
    (
        "Sentiment Analysis",
        "Review Summarization",
        "Question Answering"
    )
)

# -----------------------------------
# Sentiment Analysis
# -----------------------------------

if option == "Sentiment Analysis":

    st.header("🎭 Sentiment Analysis")

    review = st.text_area(
        "Enter Movie Review",
        height=200
    )

    if st.button("Analyze Sentiment"):

        if review.strip() == "":
            st.warning("Please enter a review.")

        else:

            result = sentiment_model(review)[0]

            st.success("Prediction Complete")

            st.write("### Sentiment")
            st.write(result["label"])

            st.write("### Confidence")
            st.write(f"{result['score']*100:.2f}%")

# -----------------------------------
# Summarization
# -----------------------------------

elif option == "Review Summarization":

    st.header("📝 Review Summarization")

    review = st.text_area(
        "Paste Long Review",
        height=250
    )

    if st.button("Generate Summary"):

        if review.strip() == "":
            st.warning("Please enter review.")

        else:

            summary = summary_model(
                review,
                max_length=120,
                min_length=30,
                do_sample=False
            )

            st.success("Summary Generated")

            st.write(summary[0]["summary_text"])

# -----------------------------------
# Question Answering
# -----------------------------------

else:

    st.header("❓ Ask Questions from Review")

    context = st.text_area(
        "Paste Movie Review",
        height=250
    )

    question = st.text_input(
        "Ask a Question"
    )

    if st.button("Get Answer"):

        if context == "" or question == "":
            st.warning("Please enter both review and question.")

        else:

            answer = qa_model(
                question=question,
                context=context
            )

            st.success("Answer")

            st.write(answer["answer"])

            st.write(
                f"Confidence: {answer['score']:.2f}"
            )
