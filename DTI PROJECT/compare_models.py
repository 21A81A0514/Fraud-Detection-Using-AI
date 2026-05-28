import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

sample = "Winton arranged eight trains, known as the Kindertransports (children's transports), to evacuate the children, and died on the anniversary of the 1939 departure of the one carrying the largest number of children: 241. Winton was knighted by Queen Elizabeth II in 2003 for his efforts, despite keeping it secret for nearly 50 years."

df = pd.read_csv('fake_or_real_news.csv')
X = df['text']
y = df['label'].map({'FAKE':0, 'REAL':1})
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
models = [
    ('NB', MultinomialNB()),
    ('LR', LogisticRegression(max_iter=1000)),
    ('SVC', LinearSVC(max_iter=10000)),
    ('RF', RandomForestClassifier(n_estimators=100, random_state=42))
]
for name, clf in models:
    pipe = Pipeline([
        ('tfidf', TfidfVectorizer(stop_words='english', ngram_range=(1,2), max_features=10000, max_df=0.85, min_df=2)),
        ('clf', clf)
    ])
    pipe.fit(X_train, y_train)
    y_pred = pipe.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    p = pipe.predict([sample])[0]
    print(name, 'acc', acc, 'sample_pred', p)
