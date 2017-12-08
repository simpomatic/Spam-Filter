import os
import numpy as np
from collections import Counter
from sklearn.naive_bayes import MultinomialNB, GaussianNB, BernoulliNB
from sklearn.svm import SVC, NuSVC, LinearSVC
from sklearn.metrics import confusion_matrix
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer

#Seperate the label and the message
emails = [line.rstrip() for line in open('./spam_train.txt')] 
label = [line[0] for line in emails]
message = [line[2:] for line in emails]

#Data to vectors
count_vect = CountVectorizer(min_df=30)
X_train_counts = count_vect.fit_transform(message) #Tokenizes text
tfidf_transformer = TfidfTransformer()
X_train_ifidr = tfidf_transformer.fit_transform(X_train_counts) #Term Frequencies times Inverse Document Frequency

#Train the classifier
MultiNB_spam_filter = MultinomialNB().fit(X_train_ifidr, label)
Guass_spam_filter = GaussianNB().fit(X_train_ifidr.toarray(), label)
Bernoulli_spam_filter = BernoulliNB().fit(X_train_ifidr, label)
LinearSVC_spam_filter = LinearSVC().fit(X_train_ifidr, label)

#Seperate label and the message
test_emails = [line.rstrip() for line in open('./spam_test.txt')]
test_labels = [line[0] for line in test_emails]
test_message = [line[2:] for line in test_emails]

#Predictions
X_new_counts = count_vect.transform(test_message) #Tokenizes text
X_new_tfidf = tfidf_transformer.fit_transform(X_new_counts)

MultiNB_predicted = MultiNB_spam_filter.predict(X_new_tfidf)
Gaussian_predicted = Guass_spam_filter.predict(X_new_tfidf.toarray())
Bernoulli_predicted = Bernoulli_spam_filter.predict(X_new_tfidf)
LinearSVC_predicted = LinearSVC_spam_filter.predict(X_new_tfidf)

#Results
print(np.mean(MultiNB_predicted == test_labels))

print(confusion_matrix(test_labels,MultiNB_predicted))

print(np.mean(Gaussian_predicted == test_labels))

print(confusion_matrix(test_labels,Gaussian_predicted))

print(np.mean(Bernoulli_predicted == test_labels))

print(confusion_matrix(test_labels,Bernoulli_predicted))

print(np.mean(LinearSVC_predicted == test_labels))

print(confusion_matrix(test_labels,LinearSVC_predicted))