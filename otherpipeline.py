from transformers import pipeline

# 1. Sentiment Analysis
sentiment = pipeline("sentiment-analysis")
print("Sentiment Analysis:")
print(sentiment("I love using Hugging Face!"))

# 2. Text Classification (custom labels)
classifier = pipeline(
    "text-classification",
    model="distilbert-base-uncased-finetuned-sst-2-english"
)
print("\nText Classification:")
print(classifier("The new design is user-friendly and intuitive."))

# 3. Named Entity Recognition (NER)
ner = pipeline("ner", aggregation_strategy="simple")
print("\nNamed Entity Recognition:")
print(ner("Albert Einstein was born in Ulm, Germany in 1879."))

# 4. Question Answering
qa = pipeline("question-answering")
context = (
    "The Transformers library by Hugging Face provides thousands of pretrained models to perform tasks on texts "
    "such as classification, information extraction, question answering, summarization, translation, and more."
)
print("\nQuestion Answering:")
print(qa({
    'question': "What does the Transformers library provide?",
    'context': context
}))

# 5. Translation (English to French)
translator = pipeline("translation_en_to_fr")
print("\nTranslation (English to French):")
print(translator("How are you today?"))

# 6. Summarization
summarizer = pipeline("summarization")
long_text = (
    "Machine learning is a field of artificial intelligence that uses statistical techniques to give computers "
    "the ability to learn from data, without being explicitly programmed. It is seen as a subset of artificial intelligence."
)
print("\nSummarization:")
print(summarizer(long_text, max_length=50, min_length=25))

# 7. Fill-Mask (using correct mask token for BERT)
fill_mask = pipeline("fill-mask", model="bert-base-uncased")
print("\nFill-Mask:")
print(fill_mask("Hugging Face is creating a tool that the community can [MASK]."))

# 8. Zero-Shot Classification
zeroshot = pipeline("zero-shot-classification")
text = "I had a wonderful trip to the mountains!"
candidate_labels = ["travel", "cooking", "fitness"]
print("\nZero-Shot Classification:")
print(zeroshot(text, candidate_labels))
