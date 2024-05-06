import matplotlib.pyplot as plt
import pandas as pd

from train_valid_test_loader import load_train_valid_test_datasets

def create_vectors(train_tuple, valid_tuple, n_users, n_items):

    #get list of users
    user_list_train = list(train_tuple[0])
    user_list_valid = list(valid_tuple[0])

    user_list = list(user_list_train + user_list_valid)

    #get list of items
    item_list_train = list(train_tuple[1])
    item_list_valid = list(valid_tuple[1])

    item_list = list(item_list_train + item_list_valid)

    #get information from dataframes
    user_info = pd.read_csv('../data_movie_lens_100k/user_info.csv')
    movie_info = pd.read_csv('../data_movie_lens_100k/movie_info.csv')
    
    #get user ages and genders
    user_ages = []
    user_genders = []
    for user_id in user_list:
        user_ages.append( user_info[user_info['user_id'] == user_id]['age'].iloc[0])
        user_genders.append( user_info[user_info['user_id'] == user_id]['is_male'].iloc[0])

    #get years movies were released
    movie_years  = []
    for movie in item_list:
        movie_years.append(movie_info[movie_info['item_id'] == movie]['release_year'].iloc[0])
       
    #get ratings and create df
    rating_list_train = list(train_tuple[2])
    rating_list_valid = list(valid_tuple[2])

    rating_list = list(rating_list_train + rating_list_valid)

    #
    binary_rating_list = [1 if rating >= 4.5 else 0 for rating in rating_list]


    #create dataframe
    data = pd.DataFrame({
        'user': user_list,
        'item': item_list,
        'user_age': user_ages,
        'user_gender': user_genders,
        'movie_year': movie_years,
        'rating': rating_list,
        'binary_rating': binary_rating_list
    })    

    #save data to csv for later use
    data.to_csv('data.csv', index=False)
    
    return data

#create vectors for testing data to submit to leaderboard
def create_vectors_test_data(data):

    #get lists of users and items
    user_list = list(data['user_id'])
    item_list = list(data['item_id'])

    #get information from dataframes
    user_info = pd.read_csv('../data_movie_lens_100k/user_info.csv')
    movie_info = pd.read_csv('../data_movie_lens_100k/movie_info.csv')
    
    #get user ages and genders
    user_ages = []
    user_genders = []
    for user_id in user_list:
        user_ages.append( user_info[user_info['user_id'] == user_id]['age'].iloc[0])
        user_genders.append( user_info[user_info['user_id'] == user_id]['is_male'].iloc[0])

    #get years movies were released
    movie_years  = []
    for movie in item_list:
        movie_years.append(movie_info[movie_info['item_id'] == movie]['release_year'].iloc[0])
       
    #get list of ratings

    rating_list = list(data['rating'])

    #
    binary_rating_list = [1 if rating >= 4.5 else 0 for rating in rating_list]


    #create dataframe
    data = pd.DataFrame({
        'user': user_list,
        'item': item_list,
        'user_age': user_ages,
        'user_gender': user_genders,
        'movie_year': movie_years,
        'rating': rating_list,
        'binary_rating': binary_rating_list
    })    
    
    return data


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

# # Split the dataset into training and testing sets
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# # Initialize and train the logistic regression model
# model = LogisticRegression(max_iter=1000, random_state=42)
# model.fit(X_train, y_train)

# # Predict probabilities for the test set
# y_probs = model.predict_proba(X_test)[:, 1]  # get the probability of the positive class

# # Evaluate the model using the AUC-ROC score
# auc_score = roc_auc_score(y_test, y_probs)
# print("AUC-ROC Score:", auc_score)


def fit_predict_logistic(data):

    #identify feature and target variable
    X = data.iloc[:, :-1]  # feature vectors
    y = data.iloc[:, -1]   # labels

    #split into train and test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
  
    #build model
    model = LogisticRegression(max_iter=1000, random_state=42)

    #fit model to train data
    model.fit(X_train, y_train)

    #make predictions
    y_pred = model.predict(X_test)  # get the probability of the positive class

    # Evaluate the model using the AUC-ROC score
    auc_score = roc_auc_score(y_test, y_pred)
    print("AUC-ROC Score:", auc_score)

#random forest model
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection import train_test_split

def grid_search_rf(data):

    X = data.iloc[:, :-2]  # feature vectors
    y = data.iloc[:, -2]   # labels

    # Split the data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    param_grid = {
    'n_estimators': [100, 150, 200],  
    'max_depth': [None, 10, 15, 20],    
    'min_samples_split': [5, 10, 15],   
    'min_samples_leaf': [2, 4, 6]      
    }   
    dummy_grid = {
    'n_estimators': [100]     
    }   

    # Create a Random Forest classifier
    rf_classifier = RandomForestClassifier(random_state=42)

    # Initialize GridSearch
    grid_search = GridSearchCV(estimator=rf_classifier, param_grid=param_grid, cv=5, n_jobs=-1)

    # Fit the grid search to the training data
    grid_search.fit(X_train, y_train)

    # Print best parameters found
    print("Best parameters:", grid_search.best_params_)

    # Evaluate the best model on the test set
    test_score = grid_search.score(X_test, y_test)
    print("Test score:", test_score)

    # Return the best model
    return grid_search.best_estimator_

#this function takes in a randomforest model and some data, and outputs the AUC-ROC score on that data.
def fit_predict_rf(model, data):
    
    #identify feature and target variable
    X = data.iloc[:, :-2]  # feature vectors
    y = data.iloc[:, -2]   # labels

    #split into train and test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
  
    #fit model to train data
    model.fit(X_train, y_train)

    #make predictions
    y_pred = model.predict(X_test)  # get the probability of the positive class
    y_pred_proba = model.predict_proba(X_test)[:,-1]

    # Map predictions to binary ratings
    y_test_binary = [1 if rating > 4.5 else 0 for rating in y_test]
    y_pred_binary = [1 if rating > 4.5 else 0 for rating in y_pred]

    # Evaluate the model using the AUC-ROC score
    auc_score = roc_auc_score(y_test_binary, y_pred_proba)
    print("AUC-ROC Score:", auc_score)

#this function helps get the features for the test data to submit to the leaderboard
def vectorize_test():
    #grab testing data
    data = pd.read_csv('../data_movie_lens_100k/ratings_masked_leaderboard_set.csv')

    #create feature vectors for data
    data = create_vectors_test_data(data)
    print('feature vectors created')

    return data

#this function helps generate our final predictions, it takes in a model and
#writes binary predictions to a file
def predict_test(model):

    #get vectorized test data
    data = vectorize_test()

    #extract model features
    X = data.iloc[:, :-2]  # feature vectors

    #make probabilistic predictions for being part of class 5
    y_pred = model.predict_proba(X)[:,-1]

    #convert to binary (no need for casting to binary)
    #y_pred_binary = [1 if rating > 4.5 else 0 for rating in y_pred]

    with open('predicted_ratings_leaderboard.txt', 'w') as file:
    # Write each prediction to the file
        for prediction in y_pred:
            file.write(str(prediction) + '\n')





import time
if __name__ == '__main__':

    start_time = time.time()


    # Load the dataset
    train_tuple, valid_tuple, test_tuple, n_users, n_items = \
        load_train_valid_test_datasets()
    
    #NOTE: Only use one of the following two lines
    #create data vectors
    # data = create_vectors(train_tuple, valid_tuple, n_users, n_items)

    #load data from csv file
    data = pd.read_csv('./data.csv')


    #NOTE: Only use one of the following two lines
    #perform grid search to get best model
    model = grid_search_rf(data)
    print('Finished grid search!')

    #load saved best model from last hyperparameter search
    # model = RandomForestClassifier(max_depth =  15, min_samples_leaf = 2, min_samples_split = 15, n_estimators =  200)

    fit_predict_rf(model, data)

    end_time = time.time()
    elapsed_time = end_time - start_time
    print('Elapsed time:', elapsed_time)

    #NOTE: We are currently only fitting on train set, make sure we fit on everything
    predict_test(model)





