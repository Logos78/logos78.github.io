# -*- coding: utf-8 -*-
"""
Created on Sat Mar 21 12:49:38 2020

@author: Geoffrey
"""
from sklearn.decomposition import PCA
def plot_2d_space(X, y):   
    pca = PCA(n_components=2)
    X_disp = pca.fit_transform(X)
    colors = ['#1F77B4', '#FF7F0E']
    markers = ['o', 's']
    for l, c, m in zip(np.unique(y), colors, markers):
        plt.scatter(
            X_disp[y==l, 0],
            X_disp[y==l, 1],
            c=c, label=l, marker=m
        )
    plt.title('Imbalanced dataset (2 PCA components)')
    plt.legend(loc='upper right')
    plt.show()


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn import preprocessing
import imblearn
import seaborn as sns
#%matplotlib inline
np.random.seed(5)


### Import data
df = pd.read_csv('C:\\Users\\Geoffrey\\Desktop\\Cours\\MOOC - IBM AI Engineer\\loan_train.csv')
df1 = pd.read_csv('C:\\Users\\Geoffrey\\Desktop\\Cours\\MOOC - IBM AI Engineer\\loan_test.csv')
df = pd.concat([df, df1])

df['due_date'] = pd.to_datetime(df['due_date'])
df['effective_date'] = pd.to_datetime(df['effective_date'])



### Visualization
# Principal by Gender
bins = np.linspace(df.Principal.min(), df.Principal.max(), 10)
g = sns.FacetGrid(df, col = "Gender", hue="loan_status", palette="Set1", col_wrap=2)
g.map(plt.hist, 'Principal', bins=bins, ec="k")
g.axes[-1].legend()
plt.show()

# Age by Gender
bins = np.linspace(df.age.min(), df.age.max(), 10)
g = sns.FacetGrid(df, col="Gender", hue="loan_status", palette="Set1", col_wrap=2)
g.map(plt.hist, 'age', bins=bins, ec="k")
g.axes[-1].legend()
plt.show()

# Day of week by Gender
df['dayofweek'] = df['due_date'].dt.dayofweek
bins = np.linspace(df.dayofweek.min(), df.dayofweek.max(), 10)
g = sns.FacetGrid(df, col="Gender", hue="loan_status", palette="Set1", col_wrap=2)
g.map(plt.hist, 'dayofweek', bins=bins, ec="k")
g.axes[-1].legend()
plt.show()

df.groupby(['Gender'])['loan_status'].value_counts(normalize=True)
df.groupby(['education'])['loan_status'].value_counts(normalize=True)
df.groupby(['Principal'])['loan_status'].value_counts(normalize=True)
df.Principal.value_counts()


### Pre-processing
# Creation of weekend variable
df['weekend'] = df['dayofweek'].apply(lambda x: 1 if (x>3)  else 0)

# Convert categorical features to numerical values
df['Gender'].replace(to_replace=['male','female'], value=[0,1],inplace=True)



### Feature selection
#final_df = df[['loan_status','Principal','terms','age','Gender','weekend']]
#final_df = pd.concat([final_df,pd.get_dummies(df['education'])], axis=1)
#final_df.drop(['Master or Above'], axis = 1,inplace=True)

final_df = df[['loan_status','age','Gender','weekend']]

### Under-sampling
# Random under sampling
df_us = final_df
class_0 = df_us[df_us.loan_status == 'PAIDOFF'].index
rand_dis = np.random.uniform(0,1,df_us.loan_status.value_counts()[0])
ratio = df_us.loan_status.value_counts()[1]/df_us.loan_status.value_counts()[0]
df_us = df_us.drop(index=class_0[rand_dis > ratio])
final_df = df_us



# Finalize dataset
X = final_df.drop(columns = ['loan_status'])
X = preprocessing.StandardScaler().fit(X).transform(X)
y = final_df['loan_status'].values



######################
### Classification ###
######################
from sklearn import metrics
from sklearn.metrics import jaccard_similarity_score
from sklearn.metrics import f1_score
from sklearn.metrics import log_loss
from sklearn.metrics import classification_report


### Sampling
# Split between train and test sets
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split( X, y, test_size=0.2, random_state=4)

## Under-sampling
# Random under-sampling (imblearn)
from imblearn.under_sampling import RandomUnderSampler
sampler = RandomUnderSampler(return_indices=True)
X_train, y_train, id = sampler.fit_sample(X_train, y_train)
plot_2d_space(X_train, y_train)

# Tomek links
from imblearn.under_sampling import TomekLinks
tl = TomekLinks(return_indices=True, ratio='majority')
X_train, y_train, id_tl = tl.fit_sample(X_train, y_train)
plot_2d_space(X_train, y_train)


## Over-sampling
# Random over-sampling (imblearn)
from imblearn.over_sampling import RandomOverSampler
sampler = RandomOverSampler(return_indices=True)
X_train, y_train, id = sampler.fit_sample(X_train, y_train)
plot_2d_space(X_train, y_train)

# SMOTE
from imblearn.over_sampling import SMOTE
smote = SMOTE(ratio='minority')
X_train, y_train = smote.fit_sample(X_train, y_train)

from imblearn.over_sampling import BorderlineSMOTE
smote = BorderlineSMOTE(ratio='minority', kind = 'borderline-1')
X_train, y_train = smote.fit_sample(X_train, y_train)

from imblearn.over_sampling import KMeansSMOTE
smote = KMeansSMOTE(random_state=0)
X_train, y_train = smote.fit_sample(X_train, y_train)

from imblearn.over_sampling import SVMSMOTE
smote = SVMSMOTE(random_state=0)
X_train, y_train = smote.fit_sample(X_train, y_train)

plot_2d_space(X_train, y_train)



# ADASYN
from imblearn.over_sampling import ADASYN
adasyn = ADASYN(ratio='minority')
X_train, y_train = adasyn.fit_sample(X_train, y_train)
plot_2d_space(X_train, y_train)


# Split between train and cross-validation sets
X_train, X_cv, y_train, y_cv = train_test_split( X_train, y_train, test_size=0.2, random_state=4)



### KNN
from sklearn.neighbors import KNeighborsClassifier
K = 10
accuracy = np.zeros((K-1))

for n in range(1,K):
    #Train model
    KNN = KNeighborsClassifier(n_neighbors = n)
    KNN.fit(X_train,y_train)
    
    # Predict
    yhat_KNN = KNN.predict(X_cv)
    accuracy[n-1] = metrics.accuracy_score(y_cv, yhat_KNN)

# Display results
plt.plot(range(1,K),accuracy,'g')
plt.legend('Accuracy')
plt.ylabel('Accuracy ')
plt.xlabel('Number of Neighbours (K)')
plt.tight_layout()
plt.show()
print( "KNearestNeighbour's accuracy (with k =", accuracy.argmax()+1, ") :", accuracy.max())

# Train model with the best k
k_KNN = accuracy.argmax()+1
KNN = KNeighborsClassifier(n_neighbors = k_KNN)
KNN.fit(X_train,y_train)
yhat_KNN = KNN.predict(X_test)
Jaccard_KNN = metrics.jaccard_similarity_score(y_test, yhat_KNN)
F1Score_KNN = f1_score(y_test, yhat_KNN, average='weighted')
#print(yhat_KNN)
print (classification_report(y_test, yhat_KNN))



### Decision Tree
# Train model
from sklearn.tree import DecisionTreeClassifier
K = 20
accuracy = np.zeros((K-3))

for depth in range(3,K) :
    # Train model
    LoanTree = DecisionTreeClassifier(criterion="entropy", max_depth = depth)
    LoanTree.fit(X_train,y_train)
    
    # Predict
    yhat_Tree = LoanTree.predict(X_cv)
    accuracy[depth-3] = metrics.accuracy_score(y_cv, yhat_Tree)

# Display results
plt.plot(range(3,K),accuracy,'g')
plt.legend('Accuracy')
plt.ylabel('Accuracy ')
plt.xlabel('Max depth')
plt.tight_layout()
plt.show()
print( "Decision Tree's accuracy (with max depth =", accuracy.argmax()+3, ") :", accuracy.max())

# Train model with the best max_depth
max_depth = accuracy.argmax()+1
LoanTree = DecisionTreeClassifier(criterion="entropy", max_depth = max_depth)
LoanTree.fit(X_train,y_train)
yhat_Tree = LoanTree.predict(X_test)
Jaccard_Tree = metrics.jaccard_similarity_score(y_test, yhat_Tree)
F1Score_Tree = f1_score(y_test, yhat_Tree, average='weighted')
#print(yhat_Tree)
print (classification_report(y_test, yhat_Tree))



### SVM
# Train model
from sklearn import svm
from sklearn.model_selection import GridSearchCV
SVM = svm.SVC()
param_grid_SVM = [{'C': [0.01, 0.1, 0.3, 1, 10], 'gamma': [0.001], 'kernel': ['linear', 'rbf', 'sigmoid']}]

gridSVM=GridSearchCV(SVM, param_grid=param_grid_SVM, cv=5)
gridSVM.fit(X_train,y_train)
print('Accuracy :', gridSVM.best_score_)
SVM_params = gridSVM.best_params_
print('Best parameters :', gridSVM.best_params_)

# Train model with best parameters
SVM = svm.SVC(C = SVM_params['C'], gamma = SVM_params['gamma'], kernel = SVM_params['kernel'])
SVM.fit(X_train, y_train)
yhat_SVM = SVM.predict(X_test)
Jaccard_SVM = jaccard_similarity_score(y_test, yhat_SVM)
F1Score_SVM = f1_score(y_test, yhat_SVM, average='weighted')
#print(yhat_SVM)
print (classification_report(y_test, yhat_SVM))



### Logistic regression
from sklearn.linear_model import LogisticRegression
LR = LogisticRegression(C=0.01, solver='liblinear')
LR.fit(X_train,y_train)

reg_parameter = [0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1]
accuracy = np.zeros(len(reg_parameter))

for i,c in enumerate(reg_parameter) :
    # Train model
    LR = LogisticRegression(C = c, solver='liblinear')
    LR.fit(X_train,y_train)
    
    # Predict
    yhat_LR = LR.predict(X_cv)
    accuracy[i] = metrics.accuracy_score(y_cv, yhat_LR)

# Display results
plt.semilogx(reg_parameter,accuracy,'g')
plt.legend('Accuracy')
plt.ylabel('Accuracy ')
plt.xlabel('C')
plt.tight_layout()
plt.show()
print( "Logistic regression's accuracy (with C =", reg_parameter[accuracy.argmax()], ") :", accuracy.max())

# Train model with the best C
C = reg_parameter[accuracy.argmax()]
LR = LogisticRegression(C = C, solver='liblinear')
LR.fit(X_train,y_train)
yhat_LR = LR.predict(X_test)
yhat_prob = LR.predict_proba(X_test)
Jaccard_LR = jaccard_similarity_score(y_test, yhat_LR)
F1Score_LR = f1_score(y_test, yhat_LR, average='weighted')
Log_LR = log_loss(y_test, yhat_prob)
#print(yhat_LR)
print (classification_report(y_test, yhat_LR))



#####################
### Final results ###
#####################
Table = pd.DataFrame()
Table['Algorithm'] = ["KNN", "Decision Tree", "SVM", "Logistic Regression"]
Table['Jaccard'] = [Jaccard_KNN, Jaccard_Tree, Jaccard_SVM, Jaccard_LR]
Table['F1-score'] = [F1Score_KNN, F1Score_Tree, F1Score_SVM, F1Score_LR]
Table['Log Loss'] = ["NA", "NA", "NA", Log_LR]
Table.head()