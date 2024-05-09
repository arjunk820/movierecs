# import matplotlib.pyplot as plt
from surprise import SVD, Dataset, Reader
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV

from train_valid_test_loader import load_train_valid_test_datasets

def tuple_to_surprise_dataset(tupl):

    ratings_dict = {
        "userID": tupl[0],
        "itemID": tupl[1],
        "rating": tupl[2],
    }

    df = pd.DataFrame(ratings_dict)

    reader = Reader(rating_scale=(1, 5))

    dataset = Dataset.load_from_df(df[["userID", "itemID", "rating"]], reader)

    return dataset

def create_vectors(train_tuple, valid_tuple, n_users, n_items, k):

    ratings_dict = {
        "userID": list(train_tuple[0]) + list(valid_tuple[0]),
        "itemID": list(train_tuple[1]) + list(valid_tuple[1]),
        "rating": list(train_tuple[2]) + list(valid_tuple[2]),
    }

    df = pd.DataFrame(ratings_dict)
    reader = Reader(rating_scale=(1, 5))
    data = Dataset.load_from_df(df[['userID', 'itemID', 'rating']], reader)
    trainset = data.build_full_trainset()

    algo = SVD(n_factors=k, random_state=42)
    algo.fit(trainset)

    user_factors = algo.pu  # User factors
    item_factors = algo.qi  # Item factors

    # Create DataFrame for user factors
    user_factors_df = pd.DataFrame(user_factors, columns=[f'user_factor_{i}' for i in range(user_factors.shape[1])])
    user_factors_df['userID'] = df['userID'].unique()

    # Create DataFrame for item factors
    item_factors_df = pd.DataFrame(item_factors, columns=[f'item_factor_{i}' for i in range(item_factors.shape[1])])
    item_factors_df['itemID'] = df['itemID'].unique()

    # Merge user factors into ratings DataFrame based on user ID
    final_df = pd.merge(df, user_factors_df, on='userID')

    # Merge item factors into final DataFrame based on item ID
    final_df = pd.merge(final_df, item_factors_df, on='itemID')

    return final_df

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

# def grid_search_rf(data):

#     X = data.iloc[:, :-2]  # feature vectors
#     y = data.iloc[:, -2]   # labels

#     # Split the data into train and test sets
#     X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#     param_grid = {
#     'n_estimators': [100, 150, 200],  
#     'max_depth': [None, 10, 15, 20],    
#     'min_samples_split': [5, 10, 15],   
#     'min_samples_leaf': [2, 4, 6]      
#     }   
#     dummy_grid = {
#     'n_estimators': [100]     
#     }   

#     # Create a Random Forest classifier
#     rf_classifier = RandomForestClassifier(random_state=42)

#     # Initialize GridSearch
#     grid_search = GridSearchCV(estimator=rf_classifier, param_grid=param_grid, cv=5,
#                                 n_jobs=-1, scoring= 'balanced_accuracy')

#     # Fit the grid search to the training data
#     grid_search.fit(X_train, y_train)

#     # Print best parameters found
#     print("Best parameters:", grid_search.best_params_)

#     # Evaluate the best model on the test set
#     test_score = grid_search.score(X_test, y_test)
#     print("Test score:", test_score)

#     # Return the best model
#     return grid_search.best_estimator_

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


#this function takes in a randomforest model and some data, and outputs the AUC-ROC score on that data.
def fit_predict_xgb(model, data):
    # Identify feature and target variable
    X = data.iloc[:, :-2]  # Feature vectors
    y = data.iloc[:, -2]   # Labels

    # Adjust class labels to start from 0
    y_adjusted = y - 1

    # Split into train and test
    X_train, X_test, y_train, y_test = train_test_split(X, y_adjusted, test_size=0.2, random_state=42)

    # Fit model to train data
    model.fit(X_train, y_train)

    # Make predictions
    y_pred_proba = model.predict_proba(X_test)[:, -1]  # Get the probability of the positive class

    # Map predictions to binary ratings
    y_test_binary = [1 if rating > 2.5 else 0 for rating in y_test]  # Threshold adjusted for 0-4 ratings
    y_pred_binary = [1 if rating > 0.5 else 0 for rating in y_pred_proba]  # Using predicted probabilities

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

    #load data
    data = pd.read_csv('./data.csv')

    X = data.iloc[:, :-2]  # Feature vectors
    y = data.iloc[:, -2]   # Labels

    #fit model on all data
    model.fit(X,y)

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




# def grid_search_xgb(data):

#     X = data.iloc[:, :-2]  # feature vectors
#     y = data.iloc[:, -2]   # labels

#     # Split the data into train and test sets
#     X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    

#     param_grid = {
#     'n_estimators': [100, 150],
#     'max_depth': [3, 5],
#     'min_child_weight': [1, 3],
#     'gamma': [0, 0.1],
#     'subsample': [0.8, 1.0],
#     'colsample_bytree': [0.8, 1.0],
#     'learning_rate': [0.05, 0.1]
# }
#     dummy_grid = {
#     'n_estimators': [100]     
#     }   

#     # Create an XGBoost classifier
#     xgb_classifier = XGBClassifier(random_state=42)

#     # Initialize GridSearch
#     grid_search = GridSearchCV(estimator=xgb_classifier, param_grid=param_grid, cv=5,
#                                n_jobs=-1, scoring='balanced_accuracy')
    
#     #adjust values to be in line with what xgb expects
#     y_train_adjusted = y_train - 1
#     y_test_adjusted = y_test - 1

#     # Fit the grid search to the adjusted training data
#     grid_search.fit(X_train, y_train_adjusted)

#     # Evaluate the best model on the adjusted test set
#     test_score = grid_search.score(X_test, y_test_adjusted)
#     print("Test score:", test_score)

#     # Return the best model
#     return grid_search.best_estimator_  

def grid_search_xgb(data):

    X = data.iloc[:, :-2]  # feature vectors
    y = data.iloc[:, -2]   # labels

    # Split the data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    

    param_grid = {
    'n_estimators': [100, 150],
    'max_depth': [3, 5],
    'min_child_weight': [1, 3],
    'gamma': [0, 0.1],
    'subsample': [0.8, 1.0],
    'colsample_bytree': [0.8, 1.0],
    'learning_rate': [0.05, 0.1]
}
    dummy_grid = {
    'n_estimators': [100]     
    }   

    # Create an XGBoost classifier
    xgb_classifier = XGBClassifier(random_state=42)

    # Initialize GridSearch
    grid_search = GridSearchCV(estimator=xgb_classifier, param_grid=param_grid, cv=5,
                               n_jobs=-1, scoring='balanced_accuracy')
    
    #adjust values to be in line with what xgb expects
    y_train_adjusted = y_train - 1
    y_test_adjusted = y_test - 1

    # Fit the grid search to the adjusted training data
    grid_search.fit(X_train, y_train_adjusted)

    # Evaluate the best model on the adjusted test set
    test_score = grid_search.score(X_test, y_test_adjusted)
    print("Test score:", test_score)

    # Return the best model
    return grid_search.best_estimator_  
from sklearn.ensemble import AdaBoostClassifier


# #ADABOOST
# def grid_search_adaboost(data):

#     X = data.iloc[:, :-2]  # feature vectors
#     y = data.iloc[:, -2]   # labels

#     # Split the data into train and test sets
#     X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#     param_grid = {
#         'n_estimators': [50, 100, 150],
#         'learning_rate': [0.1, 0.5, 1.0]
#     }

#     # Create an AdaBoost classifier
#     adaboost_classifier = AdaBoostClassifier(random_state=42)

#     # Initialize GridSearch
#     grid_search = GridSearchCV(estimator=adaboost_classifier, param_grid=param_grid, cv=5,
#                                n_jobs=-1, scoring='balanced_accuracy')

#     # Fit the grid search to the training data
#     grid_search.fit(X_train, y_train)

#     # Evaluate the best model on the test set
#     test_score = grid_search.score(X_test, y_test)
#     print("Test score:", test_score)

#     # Return the best model
#     return grid_search.best_estimator_

# def fit_predict_adaboost(model, data):
#     # Identify feature and target variable
#     X = data.iloc[:, :-2]  # Feature vectors
#     y = data.iloc[:, -2]   # Labels

#     # Adjust class labels to start from 0
#     y_adjusted = y - 1

#     # Split into train and test
#     X_train, X_test, y_train, y_test = train_test_split(X, y_adjusted, test_size=0.2, random_state=42)

#     # Fit model to train data
#     model.fit(X_train, y_train)

#     # Make predictions
#     y_pred_proba = model.predict_proba(X_test)[:, -1]  # Get the probability of the positive class

#     # Map predictions to binary ratings
#     y_test_binary = [1 if rating > 2.5 else 0 for rating in y_test]  # Threshold adjusted for 0-4 ratings
#     y_pred_binary = [1 if rating > 0.5 else 0 for rating in y_pred_proba]  # Using predicted probabilities

#     # Evaluate the model using the AUC-ROC score
#     auc_score = roc_auc_score(y_test_binary, y_pred_proba)
#     print("AUC-ROC Score:", auc_score)

def create_vectors_and_fit_models(k):


    data = create_vectors(train_tuple, valid_tuple, n_users, n_items, k=k)

    #NOTE: We should discuss if it matters whether we just make a binary prediction or make a prediction and then convert
    #add binary ratings column
    bin_ratings = [1 if rating > 4.5 else 0 for rating in data['rating']]  # Threshold adjusted for 0-4 ratings
    data['binary_rating'] = bin_ratings

    #separate features and target
    X = data.drop(['rating', 'binary_rating'], axis=1)
    y = data['binary_rating']

    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Random Forest Classifier
    #perform grid search
    rf = rf_grid_search(X,y)
    print('finished grid search')
    rf.fit(X_train, y_train)
    rf_preds = rf.predict_proba(X_test)[:, 1]
    auc_rf = roc_auc_score(y_test, rf_preds)
    auc_scores_rf.loc[k, "RF"] = auc_rf
    
    # XGBoost Classifier
    xgb = xgb_grid_search(X,y)
    print('finished xgb grid search')
    xgb.fit(X_train, y_train)
    xgb_preds = xgb.predict_proba(X_test)[:, 1]
    auc_xgb = roc_auc_score(y_test, xgb_preds)
    auc_scores_xgb.loc[k, "XGB"] = auc_xgb

#this function performs a hyperparameter grid search on the random forest model.
def rf_grid_search(X, y):
    # Define the parameter grid to search
    param_grid = {
    'n_estimators': [50, 100],
    'max_depth': [None, 10],
    'min_samples_split': [2, 5],
    'min_samples_leaf': [1, 2]
}

    # Initialize Random Forest classifier
    rf = RandomForestClassifier(random_state=42)

    # Perform grid search
    grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=5, scoring='balanced_accuracy', n_jobs=-1)
    grid_search.fit(X, y)

    # Get the best model
    best_rf = grid_search.best_estimator_

    #print best parameters
    print('random forest best parameters:', grid_search.best_params_)

    return best_rf

#this function performs a hyperparameter grid search for the xgboost model
import xgboost as xgb
def xgb_grid_search(X, y):
    # Define the parameter grid to search
    param_grid = {
        'n_estimators': [50, 100, 150],
        'max_depth': [3, 5, 7],
        'learning_rate': [0.01, 0.1, 0.3]
    }

    # Initialize XGBoost classifier
    xgb_classifier = xgb.XGBClassifier(random_state=42)

    # Perform grid search
    grid_search = GridSearchCV(estimator=xgb_classifier, param_grid=param_grid, cv=5, scoring='balanced_accuracy', n_jobs=-1)
    grid_search.fit(X, y)

    # Get the best model
    best_xgb = grid_search.best_estimator_

    #print best parameters
    print('xgb best parameters:', grid_search.best_params_)

    return best_xgb


import time
if __name__ == '__main__':

    start_time = time.time()


    # Load the dataset
    train_tuple, valid_tuple, test_tuple, n_users, n_items = \
        load_train_valid_test_datasets()
    
    auc_scores_rf = pd.DataFrame(index=[2, 10, 50], columns=["RF"])
    auc_scores_xgb = pd.DataFrame(index=[2, 10, 50], columns=["XGB"])

    for k in [2, 10, 50]:
        create_vectors_and_fit_models(k)

    # Print the AUC scores
    print("AUC Scores for Random Forest:", auc_scores_rf)
    print("AUC Scores for XGBoost:", auc_scores_xgb)

    end_time = time.time()
    elapsed_time = end_time - start_time

    print('Elapsed time:', elapsed_time)