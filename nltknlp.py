import nltk
import spacy
import spacy.cli
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk.corpus import stopwords
from nltk import pos_tag, ne_chunk
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
# from gensim.models import Word2Vec

# Ensure required downloads
# nltk.download('punkt')
# nltk.download('wordnet')
# nltk.download('averaged_perceptron_tagger')
# nltk.download('maxent_ne_chunker')
# nltk.download('words')
# nltk.download('stopwords')
# nltk.download('punkt_tab')
# nltk.download('averaged_perceptron_tagger_eng')
# nltk.download('maxent_ne_chunker_tab')
# Load spaCy model
try:
    nlp = spacy.load('en_core_web_sm')
except OSError:
    spacy.cli.download('en_core_web_sm')
    nlp = spacy.load('en_core_web_sm')

doc = """Apple is looking at buying U.K. startup for $1 billion. Natural Language Processing with NLTK and spaCy is fun. Nltk is better than spaCy for some tasks, but spaCy is faster and more efficient for others."""

# 1. Tokenization (NLTK)
tokens = word_tokenize(doc)
print("Tokens:", tokens)

# 2. Lemmatization (NLTK)
lemmatizer = WordNetLemmatizer()
lemmas = [lemmatizer.lemmatize(token) for token in tokens]
print("Lemmas:", lemmas)

# 3. Stemming (NLTK)
stemmer = PorterStemmer()
stems = [stemmer.stem(token) for token in tokens]
print("Stems:", stems)

# 4. POS Tagging (NLTK)
pos_tags = pos_tag(tokens)
print("POS Tags:", pos_tags)

# 5. Named Entity Recognition (NLTK)
ne_tree = ne_chunk(pos_tags)
print("NE Tree:")
print(ne_tree)

# 6. Stop Words Removal (NLTK)
stop_words = set(stopwords.words('english'))
filtered = [w for w in tokens if w.lower() not in stop_words]
print("Filtered Tokens:", filtered)

# 7. Chunking (NLTK)
grammar = "NP: {<DT>?<JJ>*<NN>}"
cp = nltk.RegexpParser(grammar)
chunked = cp.parse(pos_tags)
print("Chunks:")
print(chunked)

# 8. Dependency Parsing (spaCy)
print("Dependencies:")
for token in nlp(doc):
    print(f"{token.text} -> {token.dep_} -> {token.head.text}")

# 9. Text Vectorization: Bag of Words (sklearn)
bow = CountVectorizer()
bow_matrix = bow.fit_transform([doc])
print("BoW Shape:", bow_matrix.shape)
print("BoW Features:", bow.get_feature_names_out())

# 10. Text Vectorization: TF-IDF (sklearn)
tfidf = TfidfVectorizer()
tfidf_matrix = tfidf.fit_transform([doc])
print("TF-IDF Shape:", tfidf_matrix.shape)
print("TF-IDF Features:", tfidf.get_feature_names_out())

# 11. Text Vectorization: Word2Vec (gensim)
# tokens_for_w2v = [word_tokenize(sent) for sent in sent_tokenize(doc)]
# w2v_model = Word2Vec(tokens_for_w2v, vector_size=100, window=5, min_count=1, workers=2)
# print("Vector for 'Apple':", w2v_model.wv['Apple'])
