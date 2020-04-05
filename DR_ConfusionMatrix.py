
#*Confusion Matrix starts***********************************************************************************************
import math
from sklearn.metrics import classification_report, confusion_matrix

#Predicted value for the labels on entire test dataset
Y_pred = mdl.predict_generator(test_generator, N_test_examples)
y_pred = np.argmax(Y_pred, axis=1)

classes = []
number_of_examples = test_generator.n
number_of_generator_calls = N_test_examples #math.ceil(number_of_examples / (1.0 * batchsize))

#Loop over entire test dataset and get the true label values
for i in range(0,int(number_of_generator_calls)):
    classes.extend(np.array(test_generator[i][1]))

classes = np.asarray(classes).astype('int32')
#print(len(classes))

#True labels in 'classes' and 'y_pred' contains the predicted labels for the test dataset.
print('Confusion Matrix')
print(confusion_matrix(classes, y_pred))
print('Classification Report')
target_names = ['NRDR', 'RDR']
print(classification_report(classes, y_pred, target_names=target_names))
#*Confusion Matrix******************************************************************************************************
