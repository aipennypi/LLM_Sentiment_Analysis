<h1>Sentiment Analysis Using SVC and Vader</h1>
This project demonstrates a sentiment analysis pipeline that combines machine learning (Support Vector Classification with TF-IDF) and rule-based sentiment analysis (VADER). The main objective is to evaluate the performance of different approaches and compare their accuracy in classifying sentiments from textual data.

<h2>Features</h2>
Text Preprocessing:

Converts text to lowercase for uniformity.
Tokenizes text into words.
Removes stopwords for better feature representation.
Machine Learning:

Uses TF-IDF to convert text into numerical feature vectors.
Trains an SVC (Support Vector Classifier) model using TF-IDF vectors.
Rule-Based Sentiment Analysis:

Implements the VADER (Valence Aware Dictionary and sEntiment Reasoner) sentiment analyzer.
Evaluates the rule-based sentiment analysis independently.
Combined Model:

Integrates VADER's compound sentiment scores as additional features for the SVC model.
 
 Provides a Python-based sentiment analysis tool leveraging OpenAI's GPT-3.5 model. It uses the GPT model to classify the sentiment of tweets as positive, negative, or neutral. The implementation is tailored for datasets with a text column for tweet content and a sentiment column for ground truth labels.
<h2>Data</h2>
The project uses a dataset named Tweets.csv, which contains:<br>
you can download from https://www.kaggle.com/datasets/yasserh/twitter-tweets-sentiment-dataset
<br>
text: The tweet or text data to be analyzed.
sentiment: The ground truth sentiment labels (e.g., positive, negative, neutral).
Ensure the dataset is in CSV format and structured correctly before running the code.

<h2>Workflow</h2>
<h3>1. Preprocessing </h3>
Convert the text data to lowercase.
Tokenize the text into individual words using NLTK.
Remove stopwords from the tokenized words to reduce noise.
<h3>2. TF-IDF Vectorization </h3>
Use TF-IDF (Term Frequency-Inverse Document Frequency) to convert the text data into numerical feature vectors for input into the SVC model.
<h3>3. VADER Sentiment Scores</h3>
Compute VADER's compound sentiment scores for each text entry.
Use these scores both independently (for evaluation) and as additional features for the SVC model.
<h3>5. GPT</h3>
Use GPT API to get the sentiment classification
<h3>5. Training Models</h3>
Train an SVC model using only TF-IDF vectors.
Train another SVC model using a combination of TF-IDF vectors and VADER sentiment scores.



<h3>6. Evaluation</h3>
<h2>Evaluate the performance of:</h2>
1. SVC with only TF-IDF.
2. SVC with TF-IDF + VADER scores.
3. VADER sentiment analysis (rule-based).
4. GPT's sentiment analysis
Generate classification reports and accuracy scores for each approach.

<h2>Example Output</h2>
After running the script, the output includes:

Classification Report for SVC (TF-IDF):

Precision, Recall, F1-Score, and Accuracy.
Classification Report for SVC (TF-IDF + VADER):

Precision, Recall, F1-Score, and Accuracy.
Classification Report for VADER (Rule-Based):

Precision, Recall, F1-Score, and Accuracy.
<h2>How to Run</h2>
<h3>Clone the repository:</h3>
git clone https://github.com/your-username/sentiment-analysis.git
<h3>Navigate to the project directory:</h3>

cd sentiment-analysis

<h3>Ensure the Tweets.csv dataset is present in the directory.</h3>
<h3>Run the script</h3>
python sentiment_analysis.py

<h2>Results</h2>
This project demonstrates that combining machine learning and rule-based approaches can yield better sentiment analysis results than using either method independently.

