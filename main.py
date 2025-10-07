import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
def predictions(prediction):
    if 1 in prediction:
        print("THE EMAIL MIGHT BE A SPAM!!!!!")
    else:
        print("The Email is most probably not a SPAM")
def body(content):
    p=input("Enter the content or body of the email:")
    content.append(p)
data=pd.read_csv('spam mail.csv')
data['spam']=data['Category'].apply(lambda x: 1 if x=='spam' else 0)
pipe=Pipeline([
    ('vectorizer',CountVectorizer()),
    ('model',MultinomialNB())
])
X_train, X_test, y_train,y_test=train_test_split(data.Masseges,data.spam,test_size=0.2)
pipe.fit(X_train,y_train)
content=[]
body(content)
prediction=pipe.predict(content)
predictions(prediction)
print(f'The accuracy of the model: {pipe.score(X_test,y_test)*100}%')