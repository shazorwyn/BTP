import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, mean_absolute_error, mean_squared_error, confusion_matrix, \
    precision_recall_fscore_support, roc_auc_score
from tensorflow.keras import Sequential
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Dense, Conv1D, Flatten, Conv2D, MaxPooling1D,MaxPooling2D
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
import numpy as np
from sklearn.model_selection import GridSearchCV
from imblearn.over_sampling import SMOTE
from matplotlib import pyplot as plt


tf.random.set_seed(9999)

epochs_number = 10  
test_set_size = 0.1 
oversampling_flag = 0 
oversampling_percentage = 0.2 

# Definition of functions
def read_data():
    rawData = pd.read_csv('preprocessedR.csv')
    
    y = rawData[['FLAG']]
    X = rawData.drop(['FLAG', 'CONS_NO'], axis=1)

    normal_count = y[y['FLAG'] == 0].shape[0]
    fraud_count = y[y['FLAG'] == 1].shape[0]

    print('Normal Consumers:                    ', normal_count)
    print('Consumers with Fraud:                ', fraud_count)
    print('Total Consumers:                     ', y.shape[0])
    print("Classification assuming no fraud:     %.2f" % (normal_count / y.shape[0] * 100), "%")


    # columns reindexing according to dates
    X.columns = pd.to_datetime(X.columns)
    X = X.reindex(X.columns, axis=1)

    # Splitting the dataset into training set and test set
    X_train, X_test, y_train, y_test = train_test_split(X, y['FLAG'], test_size=test_set_size, random_state=0)
    print("Test set assuming no fraud:           %.2f" % (y_test[y_test == 0].count() / y_test.shape[0] * 100), "%\n")

    # Oversampling of minority class to encounter the imbalanced learning
    if oversampling_flag == 1:
        over = SMOTE(sampling_strategy=oversampling_percentage, random_state=0)
        X_train, y_train = over.fit_resample(X_train, y_train)
        print("Oversampling statistics in training set: ")
        print('Normal Consumers:                    ', y_train[y_train == 0].count())
        print('Consumers with Fraud:                ', y_train[y_train == 1].count())
        print("Total Consumers                      ", X_train.shape[0])

    return X_train, X_test, y_train, y_test

def results(y_test, prediction, model_name):
    print("Accuracy", 100 * accuracy_score(y_test, prediction))
    print("RMSE:", mean_squared_error(y_test, prediction))
    print("MAE:", mean_absolute_error(y_test, prediction))
    print("F1:", 100 * precision_recall_fscore_support(y_test, prediction)[2])
    print("AUC:", 100 * roc_auc_score(y_test, prediction))
    conf_matrix = confusion_matrix(y_test, prediction)
    fig, ax = plt.subplots(figsize=(7.5, 7.5))
    fig.dpi = 400
    ax.matshow(conf_matrix, cmap=plt.cm.Blues, alpha=0.3)
    for i in range(conf_matrix.shape[0]):
        for j in range(conf_matrix.shape[1]):
            ax.text(x=j, y=i,s=conf_matrix[i, j], va='center', ha='center', size='xx-large')
    
    plt.xlabel('Predictions', fontsize=18)
    plt.ylabel('Actuals', fontsize=18)
    plt.title('Confusion Matrix ' + model_name, fontsize=18)
    plt.savefig('Confusion_Matrix '+ model_name + '.png')
  
def logistic_regression(X_train, X_test, y_train, y_test):
    print('Logistic Regression:')    
    model = LogisticRegression(C=1000, max_iter=1000, n_jobs=-1, solver='newton-cg')
    model.fit(X_train, y_train)
    prediction = model.predict(X_test)
    results(y_test, prediction, "Logistic_Regression")


def decision_tree(X_train, X_test, y_train, y_test):
    print('Decision Tree:')
    model = DecisionTreeClassifier(random_state=0)
    model.fit(X_train, y_train)
    prediction = model.predict(X_test)
    results(y_test, prediction, "Decision_Tree")


def random_forest(X_train, X_test, y_train, y_test):
    print('Random Forest:')
    model = RandomForestClassifier(n_estimators=100, min_samples_leaf=1, max_features='sqrt',  # max_depth=10,
                                   random_state=0, n_jobs=-1)
    model.fit(X_train, y_train)
    prediction = model.predict(X_test)
    results(y_test, prediction, "Random_Forest")

def artificial_neural_network_(X_train, X_test, y_train, y_test):
    print('Artificial Neural Network:')

    # Model creation
    model = Sequential()
    model.add(Dense(1000, input_dim=1034, activation='relu'))
    model.add(Dense(100, activation='relu'))
    model.add(Dense(100, activation='relu'))
    model.add(Dense(100, activation='relu'))
    model.add(Dense(10, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))

    model.compile(loss=keras.losses.binary_crossentropy,
                  optimizer='adam',
                  metrics=['accuracy'])

    # model.fit(X_train, y_train, validation_split=0, epochs=i, shuffle=True, verbose=0)
    model.fit(X_train, y_train, validation_split=0, epochs=epochs_number, shuffle=True, verbose=1)
    temp_prediction = model.predict(X_test)
    prediction=(model.predict(X_test) > 0.5).astype("int32")
    model.summary()
    results(y_test, prediction, "ANN")



# Main 
X_train, X_test, y_train, y_test = read_data()


random_forest(X_train, X_test, y_train, y_test)
logistic_regression(X_train, X_test, y_train, y_test)
decision_tree(X_train, X_test, y_train, y_test)
artificial_neural_network_(X_train, X_test, y_train, y_test)
