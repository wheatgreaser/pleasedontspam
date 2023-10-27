import pandas as pd
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.feature_extraction.text import CountVectorizer

df = pd.read_csv('spamdata.csv')

X = df['text']
y = df['label']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=1)

cv = CountVectorizer()
features = cv.fit_transform(X_train)

model = svm.SVC()

model.fit(features, y_train)
features = cv.transform(X_test)
print(metrics.accuracy_score(y_test, model.predict(features))
print(model.predict(cv.transform(['BEST OFFER IN THE MARKET BUY NOW'])))


