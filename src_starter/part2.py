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

    X.shape

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



if __name__ == '__main__':

    # Load the dataset
    train_tuple, valid_tuple, test_tuple, n_users, n_items = \
        load_train_valid_test_datasets()
    
    data = create_vectors(train_tuple, valid_tuple, n_users, n_items)

    fit_predict_logistic(data)