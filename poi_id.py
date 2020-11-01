from __future__ import division
import sys
import pickle
import matplotlib.pyplot as plt
import numpy as np 
sys.path.append("../tools/")
from feature_format import featureFormat, targetFeatureSplit
from tester import dump_classifier_and_data
from pprint import pprint
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import *
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.feature_selection import SelectKBest
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import StratifiedShuffleSplit
import pandas as pd



### Task 1: Select what features you'll use.


target_label = 'poi'

financial_features = [
    'salary', 
    'deferral_payments', 
    'total_payments', 
    'loan_advances', 
    'bonus', 
    'restricted_stock_deferred', 
    'deferred_income', 
    'total_stock_value', 
    'expenses', 
    'exercised_stock_options', 
    'other', 
    'long_term_incentive', 
    'restricted_stock', 
    'director_fees'
]

email_features = [
    'to_messages', 
    'from_poi_to_this_person', 
    'from_messages', 
    'from_this_person_to_poi', 
    'shared_receipt_with_poi'
]

### Load the dictionary containing the dataset
with open('final_project_dataset.pkl', 'r') as data_file:
    data_dict = pickle.load(data_file)

### Convert data_dict to panda dataframe
enron = pd.DataFrame.from_dict(data_dict, orient='index')
print(enron.head())

### Data exploration

### Total number of data points
print ('Total number of data points: %d' %len(data_dict))

### Allocation across classes(POI/non-POI)
num_poi = len(enron[enron['poi'].astype(np.float32)==1])

print ('Number of person of interest: %d' %num_poi)
print ('Number of non person of interest: %d' %(len(data_dict)-num_poi))

### Number of features used
all_features = financial_features + email_features
print('Number of features: %d' %len(all_features))
print('Number of financial features: %d' %len(financial_features))
print('Number of email features: %d' %len(email_features))

### Missing values in features
def num_missing_value(feature):
    num_missing_value = 0
    for name in data_dict:
        person = data_dict[name]
        if person[feature] == 'NaN':
            num_missing_value += 1
    #print ('Number of missing value in feature %s: %d' %(feature, num_missing_value))

for feature in all_features:
    num_missing_value(feature)


### Task 2: Remove outliers


### Function to plot 2 dimensions
def Plot_2dimension(data_dict, feature_x, feature_y):
    data = featureFormat(data_dict, [feature_x, feature_y])
    for point in data:
        x = point[0]
        y = point[1]
        plt.scatter(x, y)
    plt.xlabel(feature_x)
    plt.ylabel(feature_y)
    plt.show()

### Visualise outliers by 2 dimension ploting
print(Plot_2dimension(data_dict, 'salary', 'total_payments'))

### Create list of outliers based on dimension salary
outliers = []
for key in data_dict:
    val = data_dict[key]['salary']
    if val == 'NaN':
        continue
    outliers.append((key,int(val)))

### Sort the list of outliers and print the top 1 outlier in the list
print ('Outliers in terms of salary: ')
pprint(sorted(outliers,key=lambda x:x[1],reverse=True)[:1])

### Remove the top 1 outlier: the total line
data_dict.pop('TOTAL', 0)

### Sort the list of outliers and print the 3 outliers in the list
print ('Outliers in terms of salary: ')
pprint(sorted(outliers,key=lambda x:x[1],reverse=True)[1:4])

### Print out the three persons with highest salary
print ('Print out the three Employees with highest salary:')
print (data_dict['SKILLING JEFFREY K'])
print (data_dict['LAY KENNETH L'])
print (data_dict['FREVERT MARK A'])


### Task 3: Create new feature(s)


### Store data_dict to my_dataset
my_dataset = data_dict

### Create function to compute ratio
def compute_ratio(num_poi_messages, num_total_messages):
    if num_poi_messages != 'NaN' and num_total_messages != 'NaN' and num_total_messages != 0:
        fraction = float(num_poi_messages / num_total_messages)
    else:
        fraction = 0
    return fraction

### Create new features: percent_received_email_from_poi & percent_send_email_to_poi
for name in my_dataset:
    name = my_dataset[name]
    percent_received_email_from_poi = compute_ratio(name['from_poi_to_this_person'], name['to_messages'])
    name['percent_received_email_from_poi'] = percent_received_email_from_poi
    percent_send_email_to_poi = compute_ratio(name['from_this_person_to_poi'], name['from_messages'])
    name['percent_send_email_to_poi'] = percent_send_email_to_poi

### Update all_features list with new features
all_features = [target_label] + all_features + ['percent_received_email_from_poi', 'percent_send_email_to_poi']

### Extract features and labels from dataset
my_dataset = featureFormat(my_dataset, all_features)
labels_i, features_i = targetFeatureSplit(my_dataset)

### Create function: univariate feature selection with SelectKBest
def select_k_best(k):
    select_k_best = SelectKBest(k=k)
    select_k_best.fit(features_i, labels_i)
    scores = select_k_best.scores_
    unsorted_pairs = zip(all_features[1:], scores)
    k_best_features = dict(list(reversed(sorted(unsorted_pairs, key=lambda x: x[1])))[:k])
    return [target_label] + list(k_best_features.keys())

### Create function to print out features and scores by given K value
def k_best_features_score(k):
    select_k_best = SelectKBest(k=k)
    select_k_best.fit(features_i, labels_i)
    scores = select_k_best.scores_
    unsorted_pairs = zip(all_features[1:], scores)
    k_best_features = dict(list(reversed(sorted(unsorted_pairs, key=lambda x: x[1])))[:k])
    print (k_best_features)

### Print all features and scores with SelectKBest
print ('All features and scores:')
k_best_features_score((len(all_features)-1))

### Task 4: Try a varity of classifiers


### Gaussian Naive Bayes
nb_clf = GaussianNB()
### SVC
svc_clf=SVC(probability=True)
### KNN
knn_clf = KNeighborsClassifier()
### Decision tree
dt_clf = DecisionTreeClassifier() 
### LogisticRegression
l_clf = LogisticRegression(penalty='l2')


### Task 5: Tune your classifier to achieve better than .3 precision and recall 
### using our testing script. 


### Evaluation metrics: Accuracy, precision, recall, f1
def evaluation(features, labels, clf, name):
    cv = StratifiedShuffleSplit(n_splits=10, test_size=0.3, random_state=42)
    accuracy = []
    precision = []
    recall = []
    f1 = []
    for train_index, test_index in cv.split(features,labels):
        features_train = [features[ii] for ii in train_index]
        features_test = [features[ii] for ii in test_index]
        labels_train = [labels[ii] for ii in train_index]
        labels_test = [labels[ii] for ii in test_index]
        clf.fit(features_train, labels_train)
        labels_pred = clf.predict(features_test)
        accuracy.append(round(accuracy_score(labels_test, labels_pred),2))
        precision.append(round(precision_score(labels_test, labels_pred),2))
        recall.append(round(recall_score(labels_test, labels_pred),2))
        f1.append(f1_score(labels_test, labels_pred))
    print (name)
    print ('Mean of accuracy: {0}'.format(np.mean(accuracy)))
    print ('Mean of precision: {0}'.format(np.mean(precision)))
    print ('Mean of recall: {0}'.format(np.mean(recall)))
    print ('Mean of f1 score: {0}'.format(np.mean(f1)))

### Function: GridSearchCV to tune parameters
def find_best_params(clf, features, labels, param_grid):
    grid = GridSearchCV(clf, param_grid, cv=10)
    grid.fit(features, labels)
    return grid

### Input Param Grid for KNN
k_range = list(range(1,11))
algorithm_options = ['ball_tree','kd_tree','brute','auto']
param_grid_knn = dict(n_neighbors=k_range, algorithm=algorithm_options)

### Input: Param Grid for SVC
param_grid_svc = [
  {'C': [1, 10, 50, 100, 150, 1000], 'kernel': ['linear','rbf']},
  {'C': [1, 10, 50, 100, 150, 1000], 'gamma': [0.1, 0.01, 0.001, 0.0001], 'kernel': ['linear','rbf']}]

### Input Param Grid for Decision Tree Classifier
param_grid_dt = {"criterion": ["gini", "entropy"],
              "min_samples_split": [2, 10, 20],
              "max_depth": [None, 2, 5, 10],
              "min_samples_leaf": [1, 5, 10],
              "max_leaf_nodes": [None, 5, 10, 20],
              }

### Input Param Grid for LogisticRegression Classfier
param_grid_l = {'C': [0.001, 0.01, 0.1, 1, 10, 100, 1000]}

### Function to try different k value for SelectKBest
def train_model(k):
    print ('k = {0}'.format(k))
    best_features = select_k_best(k)
    print(best_features)
    ### Save data_dict to my_dataset
    my_dataset = data_dict

    ### Extract features and labels from dataset
    my_dataset = featureFormat(my_dataset, best_features)
    labels, features = targetFeatureSplit(my_dataset)

    ### Scale features 
    features = MinMaxScaler().fit_transform(features)

    ### Find best params 
    knn_tune = find_best_params(knn_clf, features, labels, param_grid_knn)
    svc_tune = find_best_params(svc_clf, features, labels, param_grid_svc)
    dt_tune = find_best_params(dt_clf, features, labels, param_grid_dt)
    l_tune = find_best_params(l_clf, features, labels, param_grid_l)

    ### Evaluation
    evaluation(features, labels, nb_clf, 'Naive Bayes Classifier (without Tuning)')
    evaluation(features, labels, knn_clf, 'K Nearest Neighbors Classifier (without Tuning)')
    evaluation(features, labels, knn_tune, 'K Nearest Neighbors Classifier (with Tuning)')
    evaluation(features, labels, svc_clf, 'SVC Classifier (without Tuning)')
    evaluation(features, labels, svc_tune, 'SVC Classifier (with Tuning)')
    evaluation(features, labels, dt_clf, 'Decision Tree Classifier (without Tuning)')
    evaluation(features, labels, dt_tune, 'Decision Tree Classifier (with Tuning)')
    evaluation(features, labels, l_clf, 'Logistic Regression Classifier (without Tuning)')
    evaluation(features, labels, l_tune, 'Logistic Regression Classifier (with Tuning)')


### Try different k to find out the best number of features
k_best = list(range(3,10))
for k in k_best:
    train_model(k)

### With k = 4, naive bayes classfier shows the best perfoamnce.
### Print out the best features and scores
print ('Best features selected and Scores:')
k_best_features_score(4)

### Task 6: Dump your classifier, dataset, and features_list so anyone can
### check your results. You do not need to change anything below, but make sure
### that the version of poi_id.py that you submit can be run on its own and
### generates the necessary .pkl files for validating your results.

### Save the best performing classifier as clf for export
clf = nb_clf

### Save best features as features_list for export
features_list = select_k_best(4)

### Save to my_dataset for export
my_dataset = data_dict

dump_classifier_and_data(clf, my_dataset, features_list)