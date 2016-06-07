from __future__ import print_function
from sklearn.decomposition import PCA
from sklearn.datasets import fetch_mldata
from sklearn import metrics, cross_validation
from sklearn.neural_network import MLPClassifier
from sklearn.neural_network import BernoulliRBM
import matplotlib.pyplot as plt
import numpy as np
import timeit

mnist = fetch_mldata("MNIST original")

# rescale the data, use the traditional train/test split
X, y = mnist.data / 255., mnist.target

# separate train, test
X_train, X_test = X[:60000], X[60000:]
y_train, y_test = y[:60000], y[60000:]

# PCA
pca = PCA(n_components=100)
pca.fit(X_train)
np.sum(pca.explained_variance_ratio_)

plt.plot(pca.explained_variance_ratio_)
plt.title('Variance Explained by Extracted Componenent')
plt.show()

train_ext = pca.transform(X_train)
test_ext = pca.transform(X_test)


# SVM
classifier = svm.SVC(gamma=0.001)
start_time = timeit.default_timer()
classifier.fit(train_ext, y_train)
elapsed = timeit.default_timer() - start_time
print("Time: %.2f sec" % (elapsed))
predicted = classifier.predict(test_ext)

# Cross validation
start_time = timeit.default_timer()
scores = cross_validation.cross_val_score(classifier, train_ext, y_train, cv=5, n_jobs=-1)
elapsed = timeit.default_timer() - start_time
print("Time: %.2f sec" % (elapsed))
np.set_printoptions(precision=4)
print("Cross validation scores:", scores)
print("Cross validation mean: %0.3f (+/-%0.03f)" % (scores.mean(), scores.std() * 2))

# Metrics
print("Consfusion matrix:\n%s" % metrics.confusion_matrix(y_test, predicted, range(10)))
print("Accuracy on test data: %.2f" % metrics.accuracy_score(y_test, predicted))
print(metrics.classification_report(y_test, predicted))

# Show Images
misslabels = predicted[y_test != predicted]
reallabels = y_test[y_test != predicted]
missimages = X_test[y_test != predicted]
choice = np.random.choice(range(len(misslabels)), 8, replace=False)
for index, i in enumerate(choice):
    plt.subplot(2, 4, index+1)
    plt.axis('off')
    plt.imshow(missimages[i].reshape(28, 28), cmap=plt.cm.gray_r, interpolation='nearest')
    plt.title('Prediction: %i\nTrue: %i' % (misslabels[i], reallabels[i]))