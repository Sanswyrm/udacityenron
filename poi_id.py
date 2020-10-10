#!/usr/bin/python

import sys
import pickle
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
sys.path.append("../tools/")
from pprint import pprint
from feature_format import featureFormat, targetFeatureSplit
from tester import dump_classifier_and_data
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import *
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.datasets import load_iris 
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression 


import operator
### Task 1: Select what features you'll use.
### features_list is a list of strings, each of which is a feature name.
### The first feature must be "poi".

### Formatted Features list as vertical list to avoid missing commas and more easily find if any EOL erros
features_list = [
        'poi',
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
        'director_fees', 
        'to_messages',
        'from_poi_to_this_person',
        'from_messages',
        'from_this_person_to_poi',
        'shared_receipt_with_poi']                      

### Load the dictionary containing the dataset
with open('final_project_dataset.pkl', 'r') as data_file:
    data_dict = pickle.load(data_file)
    
### Convert the file to a Pandas Dataframe per mentor suggestion
    
enron = pd.DataFrame.from_dict(data_dict, orient = 'index')
print(enron.head())

### Total Number of Data Points
total_data_points = len(data_dict)
print( 'Total Number of data points: ' + str(total_data_points))

### Number of POIs
num_poi = len(enron[enron['poi'].astype(np.float32)==1])
num_non_poi = len(data_dict) - num_poi

print ('Number of POIs:' + str(num_poi))
print ('Number of Non-POIs:' + str(num_non_poi))

### Number of Features
num_features = len(features_list)
print ('Number of features: ' + str(num_features))

### Missing Features
def num_missing_value(feature):
    num_missing_value = 0
    for name in data_dict:
        person = data_dict[name]
        if person[feature] == 'NaN':
            num_missing_value += 1
    print ('Number of Missing features:' + str(num_missing_value))
    

### Task 2: Remove outliers
### Function to plot 2 dimensions
def plot_two_dimensions(data_dict, feature_x, feature_y):
     data = featureFormat(data_dict, [feature_x, feature_y])
     for point in data:
         x = point[0]
         y = point[1]
         plt.scatter(x, y)
     plt.xlabel(feature_x)
     plt.ylabel(feature_y)
     plt.show()

## Plotting 'total_payments' and 'total_stock_values'

print (plot_two_dimensions(data_dict, 'salary', 'total_payments'))
print (plot_two_dimensions(data_dict, 'salary', 'total_stock_value'))     

###total_payments outlier 
total_payment_outlier = []
for key in data_dict:
    val = data_dict[key]['total_payments']
    if val == 'NaN':
        continue
    total_payment_outlier.append((key,int(val)))


##Total Stock outlier
total_stock_value_outlier = []
for key in data_dict:
    val = data_dict[key]['total_stock_value']
    if val == 'NaN':
        continue
    total_stock_value_outlier.append((key,int(val)))



### Salary Outlier    
outliers = []
for key in data_dict:
    val = data_dict[key]['salary']
    if val == 'NaN':
        continue
    outliers.append((key,int(val)))

### Sort the list of outliers and print the top 1 outlier in the list
print ('Outliers in terms of Total Payments:')
pprint(sorted(total_payment_outlier, key =lambda x:x[1],reverse=True)[:3])

print ('Outliers in terms of Total Stock Value:' )   
pprint(sorted(total_payment_outlier, key =lambda x:x[1],reverse=True)[:3])
print ('Outliers in terms of salary: ')
pprint(sorted(outliers,key=lambda x:x[1],reverse=True)[:3])

### Remove the top 1 outlier: the total line
data_dict.pop('TOTAL')

### Reprint Plot with outlier 'Total Removed"
print (plot_two_dimensions(data_dict, 'salary', 'total_payments'))
print (plot_two_dimensions(data_dict, 'salary', 'total_stock_value'))

total_payment_outlier = []
for key in data_dict:
    val = data_dict[key]['total_payments']
    if val == 'NaN':
        continue
    total_payment_outlier.append((key,int(val)))


##Total Stock outlier
total_stock_value_outlier = []
for key in data_dict:
    val = data_dict[key]['total_stock_value']
    if val == 'NaN':
        continue
    total_stock_value_outlier.append((key,int(val)))

### Salary Outlier    
outliers = []
for key in data_dict:
    val = data_dict[key]['salary']
    if val == 'NaN':
        continue
    outliers.append((key,int(val)))


### Sort the list of outliers and print the 3 outliers in the list
print ('Outliers in terms of salary: ')
pprint(sorted(outliers,key=lambda x:x[1],reverse=True)[:3])
print ("")

print ('Outliers in terms of Total Payments: ')
pprint(sorted(total_payment_outlier, key =lambda x:x[1],reverse=True)[:3])
print ("")

print ('Outliers in terms of Total Stock Value:')    
pprint(sorted(total_payment_outlier, key =lambda x:x[1],reverse=True)[:3])
print ("")
### Task 3: Create new feature(s)
### Store to my_dataset for easy export below.
my_dataset = data_dict

#create the function to compute the percentages of emails
def percent_email(num_poi_messages, num_total_messages):
    if num_poi_messages != 'NaN' and num_total_messages != 'NaN' and num_total_messages != 0:
        fraction = float(num_poi_messages / num_total_messages)
    else: fraction = 0
    
    return fraction
    
#create the new features percent_received_from_poi, percent_sent_to_poi
for name in my_dataset:
    name = my_dataset[name]
    percent_received_from_poi = percent_email(name['from_poi_to_this_person'], name['to_messages'])
    name['percent_received_from_poi'] = percent_received_from_poi
    percent_sent_to_poi = percent_email(name['from_this_person_to_poi'], name['from_messages'])
    name['percent_sent_to_poi'] = percent_sent_to_poi
    
#update my_features list with new features
features_list = features_list + ['percent_received_from_poi', 'percent_sent_to_poi']
    
### Extract features and labels from dataset for local testing
my_dataset = featureFormat(my_dataset, features_list)
labels, features =targetFeatureSplit(my_dataset)

###Using SelectKBest to choose the best features to run through the classifiers
def k_best_features_score(k):
    select_k_best = SelectKBest(k=k)
    select_k_best.fit(features, labels)
    scores = select_k_best.scores_
    unsorted_pairs = zip(features_list[1:], scores)
    k_best = dict(list(reversed(sorted(unsorted_pairs, key=lambda x: x[1])))[:k])
    return k_best


k_best = k_best_features_score(len(features_list)-1)


sorted_dict_scores = sorted(k_best.items(), key=operator.itemgetter(1),reverse = True)
print ('All features and scores:')
print (sorted_dict_scores)
print ("") 


### Task 4: Try a varity of classifiers
### Please name your classifier clf for easy export below.
### Note that if you want to do PCA or other multi-stage operations,
### you'll need to use Pipelines. For more info:
### http://scikit-learn.org/stable/modules/pipeline.html

#
from sklearn.model_selection import train_test_split
features_train, features_test, labels_train, labels_test = \
    train_test_split(features, labels, test_size=0.3, random_state=42)

# Provided to give you a starting point. Try a variety of classifiers.

### Gaussian Naive Bayes
nb_clf = GaussianNB()
nb_clf.fit(features_train, labels_train)
nb_score = nb_clf.score(features_test, labels_test)
print ("Naive Bayes Score: " + str(nb_score))
print ("")

#Using Pipeline to Run other Classifiers as code would 'hang up' when attempting to run SVC, KNN and Decision Tree
classifiers = [('svc', SVC(C=1.0, kernel='linear')), ('tree', DecisionTreeClassifier()),('KNN', KNeighborsClassifier())]
pipe =  Pipeline(classifiers)
grid = GridSearchCV(pipe, param_grid=parameters, cv = 5)
grid.fit(features_train, labels_train)
print "score = %3.2f" %(grid.score(features_test,labels_test))
print grid.best_params_

### Task 5: Tune your classifier to achieve better than .3 precision and recall 
### using our testing script. Check the tester.py script in the final project
### folder for details on the evaluation method, especially the test_classifier
### function. Because of the small size of the dataset, the script uses
### stratified shuffle split cross validation. For more info: 
### http://scikit-learn.org/stable/modules/generated/sklearn.cross_validation.StratifiedShuffleSplit.html

# Example starting point. Try investigating other evaluation techniques!
from sklearn.model_selection import train_test_split
features_train, features_test, labels_train, labels_test = \
    train_test_split(features, labels, test_size=0.3, random_state=42)

from sklearn.model_selection import train_test_split

features_train, features_test, labels_train, labels_test = train_test_split(features, labels ,test_size=0.3, random_state = 42)








### Task 6: Dump your classifier, dataset, and features_list so anyone can
### check your results. You do not need to change anything below, but make sure
### that the version of poi_id.py that you submit can be run on its own and
### generates the necessary .pkl files for validating your results.

#dump_classifier_and_data(clf, my_dataset, features_list)