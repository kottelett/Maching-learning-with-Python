from sklearn.neural_network import MLPClassifier
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import preprocessing, cross_validation, metrics
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix

labeled_images = pd.read_csv('train.csv')
images = labeled_images.iloc[0:40000,1:]
labels = labeled_images.iloc[0:40000,:1]

train_images, test_images,train_labels, test_labels = train_test_split(images, labels, train_size=0.8, random_state=0)

clf = MLPClassifier(hidden_layer_sizes=(30, 30, 30))
clf.fit(train_images, train_labels)

predictions = clf.predict(test_images)
accuracy = clf.score(test_images, test_labels)
print(classification_report(test_labels, predictions))

cm = metrics.confusion_matrix(test_labels, predictions)
plt.figure(figsize=(9,9))
sns.heatmap(cm, annot=True, fmt=".3f", linewidths=.5, square = True, cmap = 'Blues_r')
plt.ylabel(test_labels)
plt.xlabel(predictions)
all_sample_title = 'Accuracy Score: {0}'.format(accuracy )
plt.title(all_sample_title, size = 15)
plt.show()
