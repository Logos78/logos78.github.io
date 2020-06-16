# -*- coding: utf-8 -*-
"""
Created on Sat Mar 21 12:49:38 2020

@author: Geoffrey
"""
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn import preprocessing
import imblearn
import seaborn as sns
#%matplotlib inline
np.random.seed(5)

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


### Import data
df = pd.read_csv('C:\\Users\\Geoffrey\\Desktop\\Cours\\MOOC - IBM AI Engineer\\loan_train.csv')
df1 = pd.read_csv('C:\\Users\\Geoffrey\\Desktop\\Cours\\MOOC - IBM AI Engineer\\loan_test.csv')
df = pd.concat([df, df1])

df['due_date'] = pd.to_datetime(df['due_date'])
df['effective_date'] = pd.to_datetime(df['effective_date'])
df['dayofweek'] = df['due_date'].dt.dayofweek



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
bins = np.linspace(df.dayofweek.min(), df.dayofweek.max(), 10)
g = sns.FacetGrid(df, col="Gender", hue="loan_status", palette="Set1", col_wrap=2)
g.map(plt.hist, 'dayofweek', bins=bins, ec="k")
g.axes[-1].legend()
plt.show()

# Loan status distribution
print(df.groupby(['Gender'])['loan_status'].value_counts(normalize=True), "\n")
print(df.Gender.value_counts(), "\n")

print("\n", df.groupby(['education'])['loan_status'].value_counts(normalize=True), "\n")
print(df.education.value_counts(), "\n")

print("\n", df.groupby(['Principal'])['loan_status'].value_counts(normalize=True), "\n")
print(df.Principal.value_counts(), "\n")

print("\n", df.groupby(['terms'])['loan_status'].value_counts(normalize=True), "\n")
print(df.terms.value_counts(), "\n")

print("On average, around 74% of the loans are repaid, Principal seems to be particularly irrelevant.")
print("Education and terms variable could help a little.")
print("Considering the variables, few give good explanations about loan repayment.")
print("Achieving an accuracy significantly better than 75% with this dataset is unlikely.")



### Pre-processing
# Creation of weekend variable from dayofweek variable
print(df.groupby(['dayofweek'])['loan_status'].value_counts(normalize=True),"\n")
df['weekend'] = df['dayofweek'].apply(lambda x: 1 if (x>0 and x<4)  else 0)
# Result
print(df.weekend.value_counts(), "\n")
print(df.groupby(['weekend'])['loan_status'].value_counts(normalize=True))

# Convert categorical features to numerical values
df['Gender'].replace(to_replace=['male','female'], value=[0,1],inplace=True)



### Feature selection
final_df = df[['loan_status','Principal','terms','age','Gender','weekend']]
final_df = pd.concat([final_df,pd.get_dummies(df['education'])], axis=1)
final_df.drop(['Master or Above'], axis = 1,inplace=True)

# Reduced dataset wih only the most significant variables
#final_df = df[['loan_status','age','Gender','weekend']]


# Finalize dataset
X = final_df.drop(columns = ['loan_status'])
X = preprocessing.StandardScaler().fit(X).transform(X)
y = final_df['loan_status'].values

# Split between train and test sets
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split( X, y, test_size=0.2, random_state=4)



### Sampling
# Skip this part to not use sampling technique but
# Without using sampling techniques, the model will likely to make only 'PAIDOFF' predictions
# because the wo classes are too imbalanced (75/25%)
from imblearn.under_sampling import RandomUnderSampler
from imblearn.under_sampling import TomekLinks
from imblearn.over_sampling import RandomOverSampler
from imblearn.over_sampling import SMOTE
from imblearn.over_sampling import BorderlineSMOTE
from imblearn.over_sampling import KMeansSMOTE
from imblearn.over_sampling import SVMSMOTE
from imblearn.combine import SMOTETomek
from imblearn.over_sampling import ADASYN

# You can try several method of under-sampling and over-sampling
# Execute the line below before sampling or re-sampling
# Choose your sampling method
# Then execute the line compare_models(X_train, y_train) in the console
X_train, X_test, y_train, y_test = train_test_split( X, y, test_size=0.2, random_state=4)

## Under-sampling
# Random under-sampling (imblearn)
sampler = RandomUnderSampler(return_indices=True)
X_train, y_train, id = sampler.fit_sample(X_train, y_train)

# Tomek links
tl = TomekLinks(return_indices=True, ratio='majority')
X_train, y_train, id_tl = tl.fit_sample(X_train, y_train)


## Over-sampling
# Random over-sampling (imblearn)
sampler = RandomOverSampler(return_indices=True)
X_train, y_train, id = sampler.fit_sample(X_train, y_train)

# SMOTE
smote = SMOTE(sampling_strategy = 'minority')
X_train, y_train = smote.fit_sample(X_train, y_train)

smote = BorderlineSMOTE(sampling_strategy = 'minority', kind = 'borderline-1')
X_train, y_train = smote.fit_sample(X_train, y_train)

# Not working
smote = KMeansSMOTE(sampling_strategy = 'minority', random_state=0)
X_train, y_train = smote.fit_sample(X_train, y_train)

smote = SVMSMOTE(sampling_strategy = 'minority', random_state=0)
X_train, y_train = smote.fit_sample(X_train, y_train)

smote = SMOTETomek(sampling_strategy='minority', random_state = 0)
X_train, y_train = smote.fit_sample(X_train, y_train)

# ADASYN
adasyn = ADASYN(ratio='minority')
X_train, y_train = adasyn.fit_sample(X_train, y_train)


# Final result (before / after)
plot_2d_space(X,y)
plot_2d_space(X_train, y_train)

#compare_models(X_train, y_train)
# Best results are obtained with SVMSmote sampling technique and a SVM model
# This model achieves an accuracy around 74% (with 0.75 F1-score)
# However, when you look at the recall and precision values, other techniques might be more usefull 


######################
### Classification ###
######################
from sklearn import metrics
from sklearn.metrics import jaccard_score
from sklearn.metrics import f1_score
from sklearn.metrics import log_loss
from sklearn.metrics import classification_report

from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn import svm
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression


def compare_models(X_train, y_train) :
    # Split between train and cross-validation sets
    X_train, X_cv, y_train, y_cv = train_test_split( X_train, y_train, test_size=0.2, random_state=4)
    
    ### KNN
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
#    plt.plot(range(1,K),accuracy,'g')
#    plt.legend('Accuracy')
#    plt.ylabel('Accuracy ')
#    plt.xlabel('Number of Neighbours (K)')
#    plt.tight_layout()
#    plt.show()
#    print( "KNearestNeighbour's accuracy (with k =", accuracy.argmax()+1, ") :", accuracy.max())
    
    # Train model with the best k
    k_KNN = accuracy.argmax()+1
    KNN = KNeighborsClassifier(n_neighbors = k_KNN)
    KNN.fit(X_train,y_train)
    yhat_KNN = KNN.predict(X_test)
    Jaccard_KNN = metrics.jaccard_score(y_test, yhat_KNN, pos_label='PAIDOFF')
    F1Score_KNN = f1_score(y_test, yhat_KNN, average='weighted')
    KNN_validity = sum(yhat_KNN != 'PAIDOFF')/len(yhat_KNN)
    if KNN_validity < 0.1 :
        KNN_validity = False
    else :
        KNN_validity = True
    print("KNN\n", (classification_report(y_test, yhat_KNN)))
    
    
    
    ### Decision Tree
    # Train model
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
#    plt.plot(range(3,K),accuracy,'g')
#    plt.legend('Accuracy')
#    plt.ylabel('Accuracy ')
#    plt.xlabel('Max depth')
#    plt.tight_layout()
#    plt.show()
#    print( "Decision Tree's accuracy (with max depth =", accuracy.argmax()+3, ") :", accuracy.max())
    
    # Train model with the best max_depth
    max_depth = accuracy.argmax()+1
    LoanTree = DecisionTreeClassifier(criterion="entropy", max_depth = max_depth)
    LoanTree.fit(X_train,y_train)
    yhat_Tree = LoanTree.predict(X_test)
    Jaccard_Tree = metrics.jaccard_score(y_test, yhat_Tree, pos_label='PAIDOFF')
    F1Score_Tree = f1_score(y_test, yhat_Tree, average='weighted')
    Tree_validity = sum(yhat_Tree != 'PAIDOFF')/len(yhat_KNN)
    if Tree_validity < 0.1 :
        Tree_validity = False
    else :
        Tree_validity = True
    print("Decision Tree\n", classification_report(y_test, yhat_Tree))
    
    
    
    ### SVM
    # Train model
    SVM = svm.SVC()
    param_grid_SVM = [{'C': [0.01, 0.1, 0.3, 1, 10], 'gamma': [0.001], 'kernel': ['linear', 'rbf', 'sigmoid']}]
    
    gridSVM=GridSearchCV(SVM, param_grid=param_grid_SVM, cv=5)
    gridSVM.fit(X_train,y_train)
    #print('Accuracy :', gridSVM.best_score_)
    SVM_params = gridSVM.best_params_
    #print('Best parameters :', gridSVM.best_params_)
    
    # Train model with best parameters
    SVM = svm.SVC(C = SVM_params['C'], gamma = SVM_params['gamma'], kernel = SVM_params['kernel'])
    SVM.fit(X_train, y_train)
    yhat_SVM = SVM.predict(X_test)
    Jaccard_SVM = jaccard_score(y_test, yhat_SVM, pos_label='PAIDOFF')
    F1Score_SVM = f1_score(y_test, yhat_SVM, average='weighted')
    SVM_validity = sum(yhat_SVM != 'PAIDOFF')/len(yhat_SVM)
    if SVM_validity < 0.1 :
        SVM_validity = False
    else :
        SVM_validity = True
    print("SVM\n", classification_report(y_test, yhat_SVM))
    
    
    
    ### Logistic regression
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
#    plt.semilogx(reg_parameter,accuracy,'g')
#    plt.legend('Accuracy')
#    plt.ylabel('Accuracy ')
#    plt.xlabel('C')
#    plt.tight_layout()
#    plt.show()
#    print( "Logistic regression's accuracy (with C =", reg_parameter[accuracy.argmax()], ") :", accuracy.max())
    
    # Train model with the best C
    C = reg_parameter[accuracy.argmax()]
    LR = LogisticRegression(C = C, solver='liblinear')
    LR.fit(X_train,y_train)
    yhat_LR = LR.predict(X_test)
    yhat_prob = LR.predict_proba(X_test)
    Jaccard_LR = jaccard_score(y_test, yhat_LR, pos_label='PAIDOFF')
    F1Score_LR = f1_score(y_test, yhat_LR, average='weighted')
    Log_LR = log_loss(y_test, yhat_prob)
    LR_validity = sum(yhat_LR != 'PAIDOFF')/len(yhat_LR)
    if LR_validity < 0.1 :
        LR_validity = False
    else :
        LR_validity = True
    print("Logistic regression\n", classification_report(y_test, yhat_LR))
    
    
    
    #####################
    ### Final results ###
    #####################
    Table = pd.DataFrame()
    Table['Algorithm'] = ["KNN", "Decision Tree", "SVM", "Logistic Regression"]
    Table['Jaccard'] = [Jaccard_KNN, Jaccard_Tree, Jaccard_SVM, Jaccard_LR]
    Table['F1-score'] = [F1Score_KNN, F1Score_Tree, F1Score_SVM, F1Score_LR]
    Table['Log Loss'] = ["NA", "NA", "NA", Log_LR]
    Table['Valid'] = [KNN_validity, Tree_validity, SVM_validity, LR_validity]
    
    return print("Results table\n", Table.head())