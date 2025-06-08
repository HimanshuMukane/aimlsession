from transformers import pipeline

# Initialize Hugging Face pipelines for various tasks
sentiment_pipeline = pipeline("sentiment-analysis")
opinion_pipeline = pipeline("text-classification", 
                             model="nlptown/bert-base-multilingual-uncased-sentiment")

# Example texts
texts = [
    "I absolutely love this product! It exceeded my expectations.",
    "The service was terrible and I'm never ordering again.",
    "The phone's battery life is decent, but the camera quality is amazing."
]

# Sentiment Analysis
print("--- Sentiment Analysis ---")
for txt in texts:
    result = sentiment_pipeline(txt)
    print(f"Text: {txt}\nSentiment: {result[0]['label']} (score: {result[0]['score']:.2f})\n")

# Opinion Mining / Review Analysis (multilingual star ratings)
print("--- Opinion Mining (Star Ratings) ---")
for txt in texts:
    result = opinion_pipeline(txt)
    print(f"Text: {txt}\nRating: {result[0]['label']} (score: {result[0]['score']:.2f})\n")

# Batch Processing for review analysis
def analyze_reviews(reviews):
    # reviews: list of strings
    sentiments = sentiment_pipeline(reviews)
    return [(rev, res['label'], res['score']) for rev, res in zip(reviews, sentiments)]

batch_reviews = [
    "This laptop is fantastic for gaming and work.",
    "It stopped working after a week, very disappointed.",
    "Good value for the price, battery lasts long enough."
]

print("--- Batch Review Analysis ---")
results = analyze_reviews(batch_reviews)
for rev, label, score in results:
    print(f"Review: {rev}\nSentiment: {label} (score: {score:.2f})\n")
