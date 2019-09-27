import numpy as np
import matplotlib.pyplot as plt
import itertools

from sklearn import svm, datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.utils.multiclass import unique_labels

#Confution Matrix and Classification Report
#import PlotConfusionMatrix
#import numpy as np
#Y_pred = model.predict_generator(valid_generator, num_valid_samples // batch_size+1)
#y_pred = np.argmax(Y_pred, axis=1)
#print('Confusion Matrix')
#cm = confusion_matrix(valid_generator.classes, y_pred)
#print(cm)
#print('Classification Report')
#target_names = ['Cort', 'Fender', 'Gibson', 'Ibanez', 'Jackson']
#print(classification_report(valid_generator.classes, y_pred, target_names=target_names))
#plot_confusion_matrix(cm, target_names, True)

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)
    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=0)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('Actual label')
    plt.xlabel('Predicted label')
    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    cm = np.array([[319,41],[42,258]])
    target_names = ['0', '1']
    plot_confusion_matrix(cm, target_names, True)
