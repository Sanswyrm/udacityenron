#!/usr/bin/python

import sys
import pickle
import matplotlib.pyplot
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
from sklearn.feature_selection import SelectKBest
from sklearn.metrics import accuracy_score
### Task 1: Select what features you'll use.
### features_list is a list of strings, each of which is a feature name.
### The first feature must be "poi".
features_list = ['poi','salary','from_poi_to_this_person', 'to_messages', 'from_messages'
                'from_this_person_to_poi', 'shared_receipt_with_poi'] # You will need to use more features

### Load the dictionary containing the dataset
with open("final_project_dataset.pkl", "r") as data_file:
    data_dict = pickle.load(data_file)


### Task 2: Remove outliers
### Function to plot 2 dimensions
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

### Task 3: Create new feature(s)
### Store to my_dataset for easy export below.
my_dataset = data_dict


#create the function to compute the percentages of emails
def percent_email(num_poi_messages, num_total_messages):
    if num_poi_messages != 'NaN' and num_total_messages != 'Nan' and num_total_messages != 0:
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
data = featureFormat(my_dataset, features_list, sort_keys = True)
labels, features = targetFeatureSplit(data)

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

### Task 4: Try a varity of classifiers
### Please name your classifier clf for easy export below.
### Note that if you want to do PCA or other multi-stage operations,
### you'll need to use Pipelines. For more info:
### http://scikit-learn.org/stable/modules/pipeline.html

# Provided to give you a starting point. Try a variety of classifiers.
from sklearn.naive_bayes import GaussianNB
clf = GaussianNB()

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

### Task 6: Dump your classifier, dataset, and features_list so anyone can
### check your results. You do not need to change anything below, but make sure
### that the version of poi_id.py that you submit can be run on its own and
### generates the necessary .pkl files for validating your results.

dump_classifier_and_data(clf, my_dataset, features_list)