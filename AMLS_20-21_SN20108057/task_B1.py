from result_display import display_metric_results_B
from sklearn import svm

# ======================================================================================================================
# Final Classification Model for Task B1: SVM
def model_B1(training_images, training_labels, val_images, val_labels, test_images, test_labels):
    # Fit the classifier to the training set
    clf = svm.SVC(kernel='rbf', class_weight='balanced', gamma=0.00001, C=5)
    clf.fit(training_images, training_labels)
    training_pred = clf.predict(training_images)
    val_pred = clf.predict(val_images)
    test_pred = clf.predict(test_images)
    print('Task B1')
    display_metric_results_B(training_labels, training_pred, val_labels, val_pred, test_labels, test_pred)
    print('\n\n')

    return test_pred