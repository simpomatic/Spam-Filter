import os
import numpy as np
from collections import Counter
from sklearn.naive_bayes import MultinomialNB, BernoulliNB
from sklearn.metrics import confusion_matrix
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer

#Seperate the label and the message
emails = [line.rstrip() for line in open('./spam_train.txt')]
label = [line[0] for line in emails]
message = [line[2:] for line in emails]

#Explaining what CountVectorizer does
cv1 = CountVectorizer()
print('\ncv1.fit_transform(["foo bar", "bar baz"]).A')
print("\nCountVectorizer")
cv1_fit = cv1.fit_transform(["foo bar", "bar baz"]).A
print(cv1_fit)
print("\nTerm Frequencies times Inverse Document Frequency")
tfidf_transformer = TfidfTransformer()
print(tfidf_transformer.fit_transform(cv1_fit).A)
print(cv1.get_feature_names())

#Data to vectors
#ngram_range=(1, 2)
#1-grams, “don’t,” “tase,” “me,” and “bro.”
#2-grams, “don’t tase”, “tase me”, and “me bro.”
count_vect = CountVectorizer(ngram_range=(1, 2),min_df=30)
X_train_counts = count_vect.fit_transform(message) #Tokenizes text
tfidf_transformer = TfidfTransformer()
X_train_ifidr = tfidf_transformer.fit_transform(X_train_counts) #Term Frequencies times Inverse Document Frequency

#Train the classifier
MultiNB_spam_filter = MultinomialNB().fit(X_train_ifidr, label)
Bernoulli_spam_filter = BernoulliNB().fit(X_train_ifidr, label)

MultiNB_test_predictions = MultiNB_spam_filter.predict(X_train_ifidr)
Bernoulli_test_predictions = Bernoulli_spam_filter.predict(X_train_ifidr)

#Test Accuracy
print("\nTest MultinomialNB:")

print(np.mean(MultiNB_test_predictions == label))

print(confusion_matrix(label, MultiNB_test_predictions))

print("\nTest BernoulliNB:")

print(np.mean(Bernoulli_test_predictions == label))

print(confusion_matrix(label, Bernoulli_test_predictions))

#Seperate label and the message
test_emails = [line.rstrip() for line in open('./spam_test.txt')]
test_labels = [line[0] for line in test_emails]
test_message = [line[2:] for line in test_emails]

#Predictions
X_new_counts = count_vect.transform(test_message) #Tokenizes text
X_new_tfidf = tfidf_transformer.fit_transform(X_new_counts)

MultiNB_predicted = MultiNB_spam_filter.predict(X_new_tfidf)
Bernoulli_predicted = Bernoulli_spam_filter.predict(X_new_tfidf)

#Results
print("\nCorrect Not spam\t False Not Spam")
print("\nFalse Spam\t Correct Spam")

print("\nMultinomialNB:")

print(np.mean(MultiNB_predicted == test_labels))

print(confusion_matrix(test_labels,MultiNB_predicted))

print("\nBernoulliNB:")

print(np.mean(Bernoulli_predicted == test_labels))

print(confusion_matrix(test_labels,Bernoulli_predicted))
