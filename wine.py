# -*- coding: utf-8 -*-
"""
Created on Tue May  8 13:43:12 2018

@author: lamps08
"""

# -*- coding: utf-8 -*-
"""
Created on Sun Apr 29 17:27:25 2018

@author: lamps08
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.preprocessing import StandardScaler


dataset = pd.read_csv('C:/Users/lamps08/Desktop/ml/wineq/Wine1.csv',header = 0)
dataset.head()
dataset.describe()

#checking for null values
dataset.isnull().sum()

# checking for class imbalance
values = dataset['quality'].value_counts()
print(values)

#some visualizations to understand the dataset 
sns.set_style("whitegrid")
ax = sns.countplot(x="quality", data=dataset)
ax = sns.barplot(x="quality", y= 'alcohol',data=dataset,ci=None)
ax = sns.barplot(x="quality", y= 'fixed acidity',data=dataset,ci=None)
ax = sns.barplot(x="quality", y= 'citric acid',data=dataset,ci=None)


sns.pairplot(dataset,vars = ['fixed acidity','volatile acidity','citric acid'],hue = 'quality')
sns.pairplot(dataset,vars = ['residual sugar','chlorides','free sulfur dioxide'],hue = 'quality')
sns.pairplot(dataset,vars = ['total sulfur dioxide','density','pH'],hue = 'quality')
sns.pairplot(dataset,hue = 'quality')

# Compute the correlation matrix
corr = dataset.corr()

# Generate a mask for the upper triangle
mask = np.zeros_like(corr, dtype=np.bool)
mask[np.triu_indices_from(mask)] = True

# Set up the matplotlib figure
f, ax = plt.subplots(figsize=(11, 9))

# Draw the heatmap with the mask and correct aspect ratio
sns.heatmap(corr, mask=mask, center=0,square=True, linewidths=.5) 
#corre = X.corr()
sns.plt



# class balance
from sklearn.utils import resample
# Separate majority and minority classes

df_majority = dataset[dataset.quality==6]
df_minority1 = dataset[dataset.quality==9]
df_minority2 = dataset[dataset.quality==5]
df_minority3 = dataset[dataset.quality==7]
df_minority4 = dataset[dataset.quality==8]
df_minority5 = dataset[dataset.quality==4]
df_minority6 = dataset[dataset.quality==3]
 
# Upsample minority class

df_minority1_upsampled = resample(df_minority1, 
                                 replace=True,     # sample with replacement
                                 n_samples=2198,    # to match majority class
                                 random_state=123) # reproducible results
df_minority2_upsampled = resample(df_minority2, 
                                 replace=True,     # sample with replacement
                                 n_samples=2198,    # to match majority class
                                 random_state=123) # reproducible resu
df_minority3_upsampled = resample(df_minority3, 
                                 replace=True,     # sample with replacement
                                 n_samples=2198,    # to match majority class
                                 random_state=123) # reproducible results                                 
df_minority4_upsampled = resample(df_minority4, 
                                 replace=True,     # sample with replacement
                                 n_samples=2198,    # to match majority class
                                 random_state=123) # reproducible results
df_minority5_upsampled = resample(df_minority5, 
                                 replace=True,     # sample with replacement
                                 n_samples=2198,    # to match majority class
                                 random_state=123) # reproducible results
df_minority6_upsampled = resample(df_minority6, 
                                 replace=True,     # sample with replacement
                                 n_samples=2198,    # to match majority class
                                 random_state=123) # reproducible results 
                                  
                                  
# Combine majority class with upsampled minority class
df_upsampled = pd.concat([df_majority, df_minority1_upsampled,df_minority2_upsampled,df_minority3_upsampled,df_minority4_upsampled,
                           df_minority5_upsampled,df_minority6_upsampled])
 
# Display new class counts

df_upsampled.quality.value_counts()

# sepearting features and labels
X = df_upsampled.iloc[:, :-1].values
y = df_upsampled.iloc[:, -1].values

#train and test split 

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.10, random_state=0)



# after feature scaling
sc_X = StandardScaler()
X_train_sc = sc_X.fit_transform(X_train)
X_test_sc = sc_X.transform(X_test)


#Logistic Regression

from sklearn.linear_model import LogisticRegression
regressor = LogisticRegression(multi_class='multinomial',solver = 'newton-cg')
regressor.fit(X_train_sc,y_train)
results_Train = regressor.predict(X_train_sc)
results_Test = regressor.predict(X_test_sc)

#calculate accuracy
from sklearn.metrics import accuracy_score
score_regression_test = accuracy_score(y_test,results_Test)
score_regression_train = accuracy_score(y_train,results_Train)
print("train accuracy",score_regression_train)
print ("test accuracy",score_regression_test )


#KNN

from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors = 5)
knn.fit(X_train_sc,y_train)
result_Train_knn = knn.predict(X_train_sc)
result_Test_knn = knn.predict(X_test_sc)

#calaculate accuracy
score_Knn_test = accuracy_score(y_test,result_Test_knn)
score_Knn_train = accuracy_score(y_train,result_Train_knn)
print("train accuracy",score_Knn_train)
print ("test accuracy",score_Knn_test )

from sklearn import svm
classifier = svm.SVC(kernel = 'rbf',gamma = 0.75,class_weight='balanced',probability=True)
classifier.fit(X_train_sc,y_train)
result_Train_svm = classifier.predict(X_train_sc)
result_Test_svm = classifier.predict(X_test_sc)

score_Svm_test = accuracy_score(y_test,result_Test_svm)
score_Svm_train = accuracy_score(y_train,result_Train_svm)
print("train accuracy",score_Svm_train)
print ("test accuracy",score_Svm_test )

from sklearn.ensemble import ExtraTreesClassifier
from sklearn.model_selection import GridSearchCV
param_grid = {"max_depth": [25,26,27,28,30,32],
              "max_features": [3, 5,10],
              "min_samples_split": [2, 3,7, 10],
              "min_samples_leaf": [6,7,10],
              "bootstrap": [True, False],
              "criterion": ["gini", "entropy"]}

extra_clf = ExtraTreesClassifier()
grid_search = GridSearchCV(extra_clf,param_grid=param_grid)
grid_search.fit(X_train_sc, y_train)
result_Train_et = grid_search.predict(X_train_sc)
result_Test_et = grid_search.predict(X_test_sc)

score_et_test = accuracy_score(y_test,result_Test_et)
score_et_train = accuracy_score(y_train,result_Train_et)
print("train accuracy",score_et_train)
print ("test accuracy",score_et_test )

#gradient boosting 
from sklearn.ensemble import GradientBoostingClassifier
class_gb = GradientBoostingClassifier()
class_gb.fit(X_train_sc,y_train)
result_Train_gb = class_gb.predict(X_train_sc)
result_Test_gb = class_gb.predict(X_test_sc)

score_gb_test = accuracy_score(y_test,result_Test_gb)
score_gb_train = accuracy_score(y_train,result_Train_gb)
print("train accuracy",score_gb_train)
print ("test accuracy",score_gb_test )

#mulilayer perceptreon 
from sklearn.neural_network import MLPClassifier
classifier = MLPClassifier(activation = 'tanh',solver = 'lbfgs',learning_rate = 'adaptive')
classifier.fit(X_train_sc,y_train)
result_Train_mlp = classifier.predict(X_train_sc)
result_Test_mlp = classifier.predict(X_test_sc)

score_mlp_test = accuracy_score(y_test,result_Test_mlp)
score_mlp_train = accuracy_score(y_train,result_Train_mlp)
print("train accuracy",score_mlp_train)
print ("test accuracy",score_mlp_test )

#to get the confusion matrix 
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, result_Test_mlp)
print(cm)



#### plotting confusuion matrix
class_names = list(dataset.iloc[:, -1].unique())
class_names.sort()
import itertools
from sklearn.metrics import confusion_matrix
def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

# Compute confusion matrix
cnf_matrix = confusion_matrix(y_test, result_Test_svm )
cnf_matrix2 = confusion_matrix(y_test, result_Test_mlp )
cnf_matrix3 = confusion_matrix(y_test, result_Test_gb )
cnf_matrix4 = confusion_matrix(y_test, result_Test_et )
cnf_matrix5= confusion_matrix(y_test, result_Test_knn )
cnf_matrix6 = confusion_matrix(y_test, results_Test)

np.set_printoptions(precision=2)

# Plot non-normalized confusion matrix
plt.figure()
plot_confusion_matrix(cnf_matrix, classes=class_names,
                      title='Confusion matrix SVM')
plt.show()
plot_confusion_matrix(cnf_matrix2, classes=class_names,
                      title='Confusion matrix MLP')
plt.show()
plot_confusion_matrix(cnf_matrix3, classes=class_names,
                      title='Confusion matrix Gradient Boost')
plt.show()
plot_confusion_matrix(cnf_matrix4, classes=class_names,
                      title='Confusion matrix Extra Tress')
plt.show()
plot_confusion_matrix(cnf_matrix5, classes=class_names,
                      title='Confusion matrix KNN')
plt.show()
plot_confusion_matrix(cnf_matrix6, classes=class_names,
                      title='Confusion matrix Logistic Regression')
plt.show()



