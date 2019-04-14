import numpy as np
import matplotlib.pyplot as plt, matplotlib.image as mpimg
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn import datasets
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn import svm, metrics
import seaborn as sns

labeled_images = pd.read_csv('train.csv') 
images = labeled_images.iloc[0:20000,1:]
labels = labeled_images.iloc[0:20000,:1]

train_images, test_images, train_labels, test_labels = train_test_split(images, labels, train_size=0.8, random_state=0)

#Alt som har en verdi blir en pixel med full verdi #huske å beskrive hvorfor det blir så jækla mye bedre
test_images[test_images>0]=1
train_images[train_images>0]=1

i=1
img=train_images.iloc[i].as_matrix().reshape((28,28))
plt.imshow(img ,cmap = 'binary') 
clf = svm.SVC(gamma = 0.001, C = 100) #parameters?? øøøhh
clf.fit(train_images, train_labels.values.ravel())
accuracy = clf.score(test_images, test_labels)
predictions = clf.predict(test_images)
print(accuracy)
print(test_labels)
print(classification_report(test_labels, predictions))


cm = metrics.confusion_matrix(test_labels, predictions)
plt.figure(figsize=(9,9))
sns.heatmap(cm, annot=True, fmt=".3f", linewidths=.5, square = True, cmap = 'Blues_r')
plt.ylabel(test_labels)
plt.xlabel(predictions)
all_sample_title = 'Accuracy Score: {0}'.format(accuracy)
plt.title(all_sample_title, size = 15) 
plt.show()











