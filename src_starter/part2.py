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

    #create dataframe
    data = pd.DataFrame({
        'user': user_list,
        'item': item_list,
        'user_age': user_ages,
        'user_gender': user_genders,
        'movie_year': movie_years,
        'rating': rating_list
    })    
    
    return data
    

    


if __name__ == '__main__':

    # Load the dataset
    train_tuple, valid_tuple, test_tuple, n_users, n_items = \
        load_train_valid_test_datasets()
    
    create_vectors(train_tuple, valid_tuple, n_users, n_items)