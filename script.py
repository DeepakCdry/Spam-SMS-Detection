# -*- coding: utf-8 -*-


# %% 1. Import Libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import nltk
import string
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from wordcloud import WordCloud
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import GaussianNB, MultinomialNB, BernoulliNB
from sklearn.metrics import accuracy_score, precision_score
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, BaggingClassifier, ExtraTreesClassifier, GradientBoostingClassifier, VotingClassifier, StackingClassifier
from xgboost import XGBClassifier
import pickle

# Download NLTK resources if not already present
# This block ensures that 'punkt' and 'stopwords' are available



nltk.download('punkt')       # Already downloaded, but no harm in repeating
nltk.download('punkt_tab')   # 🔺This is missing and needs to be downloaded


# Download 'punkt' if not available
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

# Download 'stopwords' if not available
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

# IMPORTANT: If you encounter an AttributeError related to 'nltk' (e.g., 'nltk' has no attribute 'download'),
# it's likely due to a circular import. This often happens if you have a file named 'nltk.py'
# in your current working directory or Python's import path.
# Please rename or delete any local 'nltk.py' file that might be conflicting with the actual NLTK library.


# %% 2. Data Loading and Initial Cleaning
# Load the dataset
# Make sure 'spam.csv' is in the same directory as this script
df = pd.read_csv("C:/Users/user/OneDrive/Desktop/internship/spam.csv", encoding='ISO-8859-1')


# Display initial shape and a sample
print("Original DataFrame shape:", df.shape)
print("Sample of original DataFrame:")
print(df.sample(5))

# Drop unnecessary columns
df.drop(columns=['Unnamed: 2', 'Unnamed: 3', 'Unnamed: 4'], inplace=True)

# Rename columns for clarity
df.rename(columns={'v1': 'target', 'v2': 'text'}, inplace=True)

# Display sample after column operations
print("\nDataFrame after dropping and renaming columns:")
print(df.sample(5))

# Encode target variable (ham: 0, spam: 1)
encoder = LabelEncoder()
df['target'] = encoder.fit_transform(df['target'])

# Check for missing values
print("\nMissing values before dropping duplicates:")
print(df.isnull().sum())

# Check for duplicate values
print("\nNumber of duplicate rows before removal:", df.duplicated().sum())

# Remove duplicate rows
df = df.drop_duplicates(keep='first')

# Verify duplicates are removed and new shape
print("Number of duplicate rows after removal:", df.duplicated().sum())
print("DataFrame shape after removing duplicates:", df.shape)

# Display value counts of the target variable
print("\nTarget variable value counts:")
print(df['target'].value_counts())

# Plotting the distribution of target variable
plt.figure(figsize=(6, 6))
plt.pie(df['target'].value_counts(), labels=['ham', 'spam'], autopct="%0.2f")
plt.title("Distribution of Ham vs. Spam Messages")
plt.show()

# %% 3. Feature Engineering (Character, Word, Sentence Counts)
# Calculate number of characters
df['num_characters'] = df['text'].apply(len)

# Calculate number of words
df['num_words'] = df['text'].apply(lambda x: len(nltk.word_tokenize(x)))

# Calculate number of sentences
df['num_sentences'] = df['text'].apply(lambda x: len(nltk.sent_tokenize(x)))

# Display head with new features
print("\nDataFrame with new character, word, and sentence count features:")
print(df.head())

# Describe statistics for ham and spam messages
print("\nDescriptive statistics for all messages:")
print(df[['num_characters', 'num_words', 'num_sentences']].describe())

print("\nDescriptive statistics for HAM messages:")
print(df[df['target'] == 0][['num_characters', 'num_words', 'num_sentences']].describe())

print("\nDescriptive statistics for SPAM messages:")
print(df[df['target'] == 1][['num_characters', 'num_words', 'num_sentences']].describe())

# Plot histograms for character counts for ham and spam
plt.figure(figsize=(12, 6))
sns.histplot(df[df['target'] == 0]['num_characters'], color='blue', label='Ham')
sns.histplot(df[df['target'] == 1]['num_characters'], color='red', label='Spam')
plt.title("Distribution of Number of Characters in Ham vs. Spam Messages")
plt.legend()
plt.show()

# Plot histograms for word counts for ham and spam
plt.figure(figsize=(12, 6))
sns.histplot(df[df['target'] == 0]['num_words'], color='blue', label='Ham')
sns.histplot(df[df['target'] == 1]['num_words'], color='red', label='Spam')
plt.title("Distribution of Number of Words in Ham vs. Spam Messages")
plt.legend()
plt.show()

# Pairplot to visualize relationships between numerical features
sns.pairplot(df, hue='target', vars=['num_characters', 'num_words', 'num_sentences'])
plt.suptitle("Pairplot of Numerical Features by Target", y=1.02)
plt.show()

# Heatmap of correlations
plt.figure(figsize=(8, 6))
sns.heatmap(df.corr(numeric_only=True), annot=True, cmap='coolwarm')
plt.title("Correlation Heatmap of Numerical Features")
plt.show()

# %% 4. Text Preprocessing (Lowercasing, Tokenization, Special Char Removal, Stopword/Punctuation Removal, Stemming)

ps = PorterStemmer()

def transform_text(text):
    text = text.lower()  # Lower case
    text = nltk.word_tokenize(text)  # Tokenization

    y = []
    for i in text:
        if i.isalnum():  # Removing special characters (keep alphanumeric)
            y.append(i)

    text = y[:]
    y.clear()

    for i in text:
        if i not in stopwords.words('english') and i not in string.punctuation:  # Removing stopwords and punctuation
            y.append(i)

    text = y[:]
    y.clear()

    for i in text:
        y.append(ps.stem(i))  # Stemming

    return " ".join(y)

# Apply the transformation to the 'text' column
df['transformed_text'] = df['text'].apply(transform_text)

# Display head with transformed text
print("\nDataFrame with transformed text:")
print(df.head())

# %% 5. WordCloud Visualization
# Generate WordCloud for ham messages

wc = WordCloud(width=800, height=400, min_font_size=10, background_color='white')

ham_wc = wc.generate(df[df['target'] == 0]['transformed_text'].str.cat(sep=" "))
# ... rest remains the same


ham_wc = wc.generate(df[df['target'] == 0]['transformed_text'].str.cat(sep=" "))
plt.figure(figsize=(10, 5))
plt.imshow(ham_wc)
plt.title("Word Cloud for Ham Messages")
plt.axis('off')
plt.show()

# Generate WordCloud for spam messages
spam_wc = wc.generate(df[df['target'] == 1]['transformed_text'].str.cat(sep=" "))
plt.figure(figsize=(10, 5))
plt.imshow(spam_wc)
plt.title("Word Cloud for Spam Messages")
plt.axis('off')
plt.show()

# %% 6. Text Vectorization (TF-IDF)
# Initialize TF-IDF Vectorizer with max_features
tfidf = TfidfVectorizer(max_features=3000)

# Fit and transform the transformed text
X = tfidf.fit_transform(df['transformed_text']).toarray()

# Get the target variable
y = df['target'].values

# Display shape of vectorized data
print("\nShape of TF-IDF vectorized data (X):", X.shape)

# %% 7. Model Building and Evaluation
# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=2)

# Initialize various classifiers
gnb = GaussianNB()
mnb = MultinomialNB()
bnb = BernoulliNB()

# Train and evaluate Naive Bayes classifiers
print("\n--- Naive Bayes Classifiers Evaluation ---")
gnb.fit(X_train, y_train)
y_pred_gnb = gnb.predict(X_test)
print("Gaussian Naive Bayes Accuracy:", accuracy_score(y_test, y_pred_gnb))
print("Gaussian Naive Bayes Precision:", precision_score(y_test, y_pred_gnb))

mnb.fit(X_train, y_train)
y_pred_mnb = mnb.predict(X_test)
print("\nMultinomial Naive Bayes Accuracy:", accuracy_score(y_test, y_pred_mnb))
print("Multinomial Naive Bayes Precision:", precision_score(y_test, y_pred_mnb))

bnb.fit(X_train, y_train)
y_pred_bnb = bnb.predict(X_test)
print("\nBernoulli Naive Bayes Accuracy:", accuracy_score(y_test, y_pred_bnb))
print("Bernoulli Naive Bayes Precision:", precision_score(y_test, y_pred_bnb))

# Initialize other classifiers
svc = SVC(kernel='sigmoid', gamma=1.0, probability=True) # probability=True needed for VotingClassifier 'soft'
knc = KNeighborsClassifier()
dtc = DecisionTreeClassifier(max_depth=5)
lrc = LogisticRegression(solver='liblinear', penalty='l1')
rfc = RandomForestClassifier(n_estimators=50, random_state=2)
abc = AdaBoostClassifier(n_estimators=50, random_state=2)
bc = BaggingClassifier(n_estimators=50, random_state=2)
etc = ExtraTreesClassifier(n_estimators=50, random_state=2)
gbdt = GradientBoostingClassifier(n_estimators=50, random_state=2)
xgb = XGBClassifier(n_estimators=50, random_state=2, use_label_encoder=False, eval_metric='logloss') # Added for XGBoost warning

clfs = {
    'SVC': svc,
    'KN': knc,
    'NB': mnb, # Multinomial Naive Bayes
    'DT': dtc,
    'LR': lrc,
    'RF': rfc,
    'AdaBoost': abc,
    'BgC': bc,
    'ETC': etc,
    'GBDT': gbdt,
    'xgb': xgb
}

# Function to train and evaluate a classifier
def train_classifier(clf, X_train, y_train, X_test, y_test):
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    return accuracy, precision

# Evaluate all classifiers
accuracy_scores = []
precision_scores = []

print("\n--- All Classifiers Evaluation ---")
for name, clf in clfs.items():
    current_accuracy, current_precision = train_classifier(clf, X_train, y_train, X_test, y_test)
    print(f"For {name}:")
    print(f"  Accuracy - {current_accuracy}")
    print(f"  Precision - {current_precision}")
    accuracy_scores.append(current_accuracy)
    precision_scores.append(current_precision)

# Create a performance DataFrame
performance_df = pd.DataFrame({
    'Algorithm': clfs.keys(),
    'Accuracy': accuracy_scores,
    'Precision': precision_scores
}).sort_values('Precision', ascending=False)

print("\nPerformance DataFrame:")
print(performance_df)

# Melt the DataFrame for plotting
performance_df1 = pd.melt(performance_df, id_vars="Algorithm")

# Plot performance comparison
plt.figure(figsize=(12, 6))
sns.catplot(x='Algorithm', y='value', hue='variable', data=performance_df1, kind='bar', height=5, aspect=2)
plt.ylim(0.5, 1.0)
plt.xticks(rotation='vertical')
plt.title("Accuracy and Precision of Different Classifiers")
plt.show()

# %% 8. Ensemble Methods (Voting and Stacking)

# Voting Classifier
print("\n--- Voting Classifier ---")
# Re-initialize classifiers for voting (SVC needs probability=True)
svc_voting = SVC(kernel='sigmoid', gamma=1.0, probability=True)
mnb_voting = MultinomialNB()
etc_voting = ExtraTreesClassifier(n_estimators=50, random_state=2)

voting = VotingClassifier(estimators=[('svm', svc_voting), ('nb', mnb_voting), ('et', etc_voting)], voting='soft')
voting.fit(X_train, y_train)
y_pred_voting = voting.predict(X_test)
print("Voting Classifier Accuracy:", accuracy_score(y_test, y_pred_voting))
print("Voting Classifier Precision:", precision_score(y_test, y_pred_voting))

# Stacking Classifier
print("\n--- Stacking Classifier ---")
# Re-initialize classifiers for stacking (SVC needs probability=True)
svc_stacking = SVC(kernel='sigmoid', gamma=1.0, probability=True)
mnb_stacking = MultinomialNB()
etc_stacking = ExtraTreesClassifier(n_estimators=50, random_state=2)

estimators = [('svm', svc_stacking), ('nb', mnb_stacking), ('et', etc_stacking)]
final_estimator = RandomForestClassifier(n_estimators=50, random_state=2)

clf = StackingClassifier(estimators=estimators, final_estimator=final_estimator)
clf.fit(X_train, y_train)
y_pred_clf = clf.predict(X_test)
print("Stacking Classifier Accuracy:", accuracy_score(y_test, y_pred_clf))
print("Stacking Classifier Precision:", precision_score(y_test, y_pred_clf))

# %% 9. Save the Model and Vectorizer
# Save the TF-IDF vectorizer and the final StackingClassifier model
# These files (vectorizer.pkl and model.pkl) are crucial for deployment

pickle.dump(tfidf, open('vectorizer.pkl', 'wb'))
pickle.dump(clf, open('model.pkl', 'wb'))




print("\n'vectorizer.pkl' and 'model.pkl' have been saved successfully.")
