import numpy as np
from sklearn import preprocessing, cross_validation, neighbors, metrics
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report

labeled_images = pd.read_csv('train.csv') 
images = labeled_images.iloc[0:500,1:]
labels = labeled_images.iloc[0:500,:1]

train_images, test_images,train_labels, test_labels = cross_validation.train_test_split(images, labels, train_size=0.8, random_state = 0)

clf = neighbors.KNeighborsClassifier()
clf.fit(train_images, train_labels.values.ravel())

accuracy = clf.score(test_images, test_labels)
print(accuracy)

test_measure = np.array([0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,41,149,156,179,254,254,201,119,46,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,13,147,241,253,253,254,253,253,253,253,245,160,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,31,224,253,253,180,174,175,174,174,174,174,223,247,145,6,0,0,0,0,0,0,0,0,0,0,0,0,7,197,254,253,165,2,0,0,0,0,0,0,12,102,184,16,0,0,0,0,0,0,0,0,0,0,0,0,152,253,254,162,18,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,235,254,158,15,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,74,250,253,15,0,0,0,16,20,19,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,7,199,253,253,0,0,25,130,235,254,247,145,6,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,20,253,253,177,100,219,240,253,253,254,253,253,125,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,5,193,253,253,254,253,253,200,155,155,238,253,229,23,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,61,249,254,241,150,30,0,0,0,215,254,254,58,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,36,39,30,0,0,0,0,0,214,253,234,31,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,41,241,253,183,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,201,253,253,102,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,114,254,253,154,5,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,62,254,255,241,30,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,10,118,235,253,249,103,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,55,81,0,102,211,253,253,253,135,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,79,243,234,254,253,253,216,117,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,48,245,253,254,207,126,27,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0])
test_measure = test_measure.reshape(1, -1)
predictions = clf.predict(test_measure)
print(predictions)
predictions = clf.predict(test_images)

print(classification_report(test_labels, predictions))


cm = metrics.confusion_matrix(test_labels, predictions)
plt.figure(figsize=(9,9))
sns.heatmap(cm, annot=True, fmt=".3f", linewidths=.5, square = True, cmap = 'Blues_r')
plt.ylabel(test_labels)
plt.xlabel(predictions)
all_sample_title = 'Accuracy Score: {0}'.format(accuracy )
plt.title(all_sample_title, size = 15)
plt.show()
