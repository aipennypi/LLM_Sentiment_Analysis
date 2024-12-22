import pandas as pd
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score
from scipy.sparse import hstack

# Initialize Sentiment Intensity Analyzer
analyzer = SentimentIntensityAnalyzer()

# Download necessary NLTK components
nltk.download('punkt_tab')       # Download tokenization tools
nltk.download('stopwords')       # Download stopwords list
nltk.download('vader_lexicon')   # Download Vader's lexicon

# Load the dataset
df = pd.read_csv('Tweets.csv')

# Convert all text to lowercase for uniformity
df['text'] = df['text'].str.lower()

# Ensure the 'text' column is of string data type
df['text'] = df['text'].astype(str)

# Tokenize the text into individual words
df['tokens'] = df['text'].apply(nltk.word_tokenize)

# Remove stopwords from the tokenized words
stopwords = nltk.corpus.stopwords.words('english')
df['tokens'] = df['tokens'].apply(lambda x: [word for word in x if word not in stopwords])

# Define features (X) and labels (y)
X = df['text']  # Text data
y = df['sentiment']  # Sentiment labels

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(X_train.shape, y_train.shape)  # Print the shape of the training set
print(X_test.shape, y_test.shape)    # Print the shape of the testing set

# Convert text data into TF-IDF vectors
vectorizer = TfidfVectorizer()
X_train_vectors = vectorizer.fit_transform(X_train)
X_test_vectors = vectorizer.transform(X_test)

# Function to compute Vader's compound sentiment score
def get_vader_score(text):
    return analyzer.polarity_scores(text)['compound']

# Compute Vader scores for training and testing sets
X_train_vader = X_train.apply(get_vader_score).values.reshape(-1, 1)
X_test_vader = X_test.apply(get_vader_score).values.reshape(-1, 1)

# Combine TF-IDF vectors with Vader compound scores
X_train_combined = hstack((X_train_vectors, X_train_vader))
X_test_combined = hstack((X_test_vectors, X_test_vader))

# Train an SVC model using only TF-IDF vectors
model = SVC()
model.fit(X_train_vectors, y_train)

# Train another SVC model using both TF-IDF vectors and Vader compound scores
model_compound = SVC()
model_compound.fit(X_train_combined, y_train)

# Function to assign sentiment based on Vader compound scores
def assign_sentiment(text):
    scores = analyzer.polarity_scores(text)
    if -0.15 < scores['compound'] < 0.15:
        return 'neutral'
    elif scores['compound'] <= -0.15:
        return 'negative'
    else:
        return 'positive'

# Predict sentiment using three approaches:
# 1. SVC with only TF-IDF vectors
y_SVC_pred = model.predict(X_test_vectors)

# 2. SVC with TF-IDF vectors + Vader compound scores
y_SVC_comp_pred = model_compound.predict(X_test_combined)

# 3. Vader's rule-based sentiment analysis
y_vad_pred = X_test.apply(assign_sentiment)
# 4. GPT sentiment analysis
import openai
apikey = os.environ.get('OPENAI_API_KEY')
# retrieve sentiment responds from chatGPT
def classify_sentiment(text):
    prompt = f"""
    Perform the following actions on the given text delimited by \
    triple backticks:
    - analyse and report the sentiment in strictly a single word \
    out of [positive, negative, neutral].

    Text to analyse: ```{text}```
    """
    client = OpenAI(api_key = apikey)
    response =  client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are a sentiment analysis assistant."},
            #{"role": "user", "content": f"Classify the sentiment of this text as positive, negative, or neutral: '{text}'"}
            #{"role": "user", "content": f"Classify the sentiment of this text as positive, negative, or neutral:'{text}'only show sentiment"}
            {"role": "user", "content": prompt}

        ]
    )
    sentiment = response.choices[0].message.content.strip().lower()
    if all(keyword not in sentiment for keyword in ['positive', 'negative', 'neutral']):      
      return 'neutral'
    if sentiment.find('positive') != -1:
        sentiments = 'positive'
    elif sentiment.find('negative') != -1:
        sentiments = 'negative'
    else:
        sentiments = 'neutral'
    return sentiments
y_GPT_pred = X.test.apply(classify_sentiment)
# Evaluate SVC with only TF-IDF vectors
print("SVC Classification Report:")
print(classification_report(y_test, y_SVC_pred))
print("Accuracy Score:", accuracy_score(y_test, y_SVC_pred))

# Evaluate SVC with TF-IDF vectors + Vader compound scores
print("SVC Compound Classification Report:")
print(classification_report(y_test, y_SVC_comp_pred))
print("Accuracy Score:", accuracy_score(y_test, y_SVC_comp_pred))
print(y_SVC_comp_pred)

# Evaluate Vader's sentiment analysis
print("Vader Classification Report:")
print(classification_report(y_test, y_vad_pred))
print("Accuracy Score:", accuracy_score(y_test, y_vad_pred))

# Evaluate GPT's sentiment analysis
print("GPT Classification Report:")
print(classification_report(y_test, y_GPT_pred))
print("Accuracy Score:", accuracy_score(y_test, y_GPT_pred))
