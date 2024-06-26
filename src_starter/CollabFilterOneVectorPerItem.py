'''
CollabFilterOneVectorPerItem.py

Defines class: `CollabFilterOneVectorPerItem`

Scroll down to __main__ to see a usage example.
'''

# Make sure you use the autograd version of numpy (which we named 'ag_np')
# to do all the loss calculations, since automatic gradients are needed
import autograd.numpy as ag_np

import matplotlib.pyplot as plt

# Use helper packages
from AbstractBaseCollabFilterSGD import AbstractBaseCollabFilterSGD
from train_valid_test_loader import load_train_valid_test_datasets

# Some packages you might need (uncomment as necessary)
## import pandas as pd
## import matplotlib
import numpy as np

# No other imports specific to ML (e.g. scikit) needed!

class CollabFilterOneVectorPerItem(AbstractBaseCollabFilterSGD):
    ''' One-vector-per-user, one-vector-per-item recommendation model.

    Assumes each user, each item has learned vector of size `n_factors`.

    Attributes required in param_dict
    ---------------------------------
    mu : 1D array of size (1,)
    b_per_user : 1D array, size n_users
    c_per_item : 1D array, size n_items
    U : 2D array, size n_users x n_factors
    V : 2D array, size n_items x n_factors

    Notes
    -----
    Inherits *__init__** constructor from AbstractBaseCollabFilterSGD.
    Inherits *fit* method from AbstractBaseCollabFilterSGD.
    '''

    def init_parameter_dict(self, n_users, n_items, train_tuple):
        ''' Initialize parameter dictionary attribute for this instance.

        Post Condition
        --------------
        Updates the following attributes of this instance:
        * param_dict : dict
            Keys are string names of parameters
            Values are *numpy arrays* of parameter values
        '''
        random_state = self.random_state # inherited RandomState object

        self.param_dict = dict(
            mu=ag_np.ones(1),
            b_per_user=ag_np.ones(n_users), # User biases
            c_per_item=ag_np.ones(n_items), # Item biases
            U=ag_np.array(0.01 * random_state.randn(n_users, self.n_factors)), # User hidden
            V=ag_np.array(0.01 * random_state.randn(n_items, self.n_factors)), # Item hidden
            )


    def predict_1(self, user_id_N, item_id_N,
                mu=None, b_per_user=None, c_per_item=None, U=None, V=None):
        ''' Predict ratings at specific user_id, item_id pairs

        Args
        ----
        user_id_N : 1D array, size n_examples
            Specific user_id values to use to make predictions
        item_id_N : 1D array, size n_examples
            Specific item_id values to use to make predictions
            Each entry is paired with the corresponding entry of user_id_N

        Returns
        -------
        yhat_N : 1D array, size n_examples
            Scalar predicted ratings, one per provided example.
            Entry n is for the n-th pair of user_id, item_id values provided.
        '''
        # Create output array
        N = user_id_N.size
        yhat_N = ag_np.ones(N)

         # Predict step
        for i in range(len(yhat_N)):
            
            #convert array boxes to np arrays
            U_np = ag_np.array(U[user_id_N[i], :])
            V_np = ag_np.array(V[item_id_N[i], :])

            # convert values to arrayboxes
            mu_arraybox = ag_np.array(mu)
            b_per_user_arraybox = ag_np.array(b_per_user[user_id_N[i]])
            c_per_item_arraybox = ag_np.array(c_per_item[item_id_N[i]])

            # Matrix factorization ratings prediction model
            try:
                dot_product = ag_np.dot(U_np, V_np)
                curr = mu_arraybox + b_per_user_arraybox + c_per_item_arraybox + dot_product

                #cast curr to an arraybox if it isn't one
                if not isinstance(curr, ag_np.numpy_boxes.ArrayBox):
                    curr = ag_np.array(curr)
                #this casting doesn't work always for some reason

                
                #get new value of yhat_N[i] depending on type of array
                if isinstance(curr, ag_np.numpy_boxes.ArrayBox):
                    yhat_N[i] = curr._value
                elif isinstance(curr, np.ndarray):
                    yhat_N[i] = curr.item()
                else:
                    yhat_N[i] = curr

            except ValueError as e:
                print(i, str(e), "Error here")
                break

        return yhat_N
    
    def predict(self, user_id_N, item_id_N,
                mu=None, b_per_user=None, c_per_item=None, U=None, V=None):
        
        inner_product = ag_np.sum(U[user_id_N] * V[item_id_N], axis = 1)
        
        res = b_per_user[user_id_N] + c_per_item[item_id_N] + mu + inner_product

        return res




    def calc_loss_wrt_parameter_dict(self, param_dict, data_tuple):
        ''' Compute loss at given parameters

        Args
        ----
        param_dict : dict
            Keys are string names of parameters
            Values are *numpy arrays* of parameter values

        Returns
        -------
        loss : float scalar
        '''
        # TIP: use self.alpha to access regularization strength
        user_ids, item_ids, y_true = data_tuple

        # Predict ratings
        y_pred = self.predict(user_ids, item_ids, **param_dict)

        # Compute the mean squared error
        mse = ag_np.sum((y_true - y_pred) ** 2)

        regularization_term_u = ag_np.sum(param_dict['U'][user_ids] ** 2)
        regularization_term_v = ag_np.sum(param_dict['V'][user_ids] ** 2)

        # Total loss: MSE + alpha * regularization
        total_loss = mse + self.alpha * (regularization_term_u + regularization_term_v)

        return total_loss 


if __name__ == '__main__':

    # Load the dataset
    train_tuple, valid_tuple, test_tuple, n_users, n_items = \
        load_train_valid_test_datasets()
    # Create the model and initialize its parameters
    # to have right scale as the dataset (right num users and items)


    # #2 factors
    # model_2 = CollabFilterOneVectorPerItem(
    #     n_epochs=150, batch_size=32, step_size=0.2,
    #     n_factors=2, alpha=0.0)
    # model_2.init_parameter_dict(n_users, n_items, train_tuple)

    # # Fit the model with SGD
    # model_2.fit(train_tuple, valid_tuple)

    # #10 factors
    # model_10 = CollabFilterOneVectorPerItem(
    #     n_epochs=150, batch_size=32, step_size=0.2,
    #     n_factors=10, alpha=0.0)
    # model_10.init_parameter_dict(n_users, n_items, train_tuple)

    # # Fit the model with SGD
    # model_10.fit(train_tuple, valid_tuple)

    # #50 factors
    # model_50 = CollabFilterOneVectorPerItem(
    #     n_epochs=150, batch_size=32, step_size=0.2,
    #     n_factors=50, alpha=0.0)
    # model_50.init_parameter_dict(n_users, n_items, train_tuple)

    # # Fit the model with SGD
    # model_50.fit(train_tuple, valid_tuple)

    # #Plot for model with 2 factors
    # plt.figure(figsize=(8, 6))
    # epoch_list_2 = model_2.trace_epoch
    # mae_list_train_2 = model_2.trace_mae_train
    # mae_list_valid_2 = model_2.trace_mae_valid
    # plt.plot(epoch_list_2, mae_list_train_2, label='Train - K=2')
    # plt.plot(epoch_list_2, mae_list_valid_2, label='Valid - K=2')
    # plt.title('MAE by epoch - K=2')
    # plt.xlabel('Epoch')
    # plt.ylabel('MAE')
    # plt.legend()
    # plt.savefig('k2_graph.png')
    # plt.close()

    # # Plot for model with 10 factors
    # plt.figure(figsize=(8, 6))
    # epoch_list_10 = model_10.trace_epoch
    # mae_list_train_10 = model_10.trace_mae_train
    # mae_list_valid_10 = model_10.trace_mae_valid
    # plt.plot(epoch_list_10, mae_list_train_10, label='Train - K=10')
    # plt.plot(epoch_list_10, mae_list_valid_10, label='Valid - K=10')
    # plt.title('MAE by epoch - K=10')
    # plt.xlabel('Epoch')
    # plt.ylabel('MAE')
    # plt.legend()
    # plt.savefig('k10_graph.png')
    # plt.close()

    # # Plot for model with 50 factors
    # plt.figure(figsize=(8, 6))
    # epoch_list_50 = model_50.trace_epoch
    # mae_list_train_50 = model_50.trace_mae_train
    # mae_list_valid_50 = model_50.trace_mae_valid
    # plt.plot(epoch_list_50, mae_list_train_50, label='Train - K=50')
    # plt.plot(epoch_list_50, mae_list_valid_50, label='Valid - K=50')
    # plt.title('MAE by epoch - K=50')
    # plt.xlabel('Epoch')
    # plt.ylabel('MAE')
    # plt.legend()
    # plt.savefig('k50_graph.png')
    # plt.close()


    #part 2 - experimenting with alpha

    # alpha_list = [0.001, 0.01, 0.1, 1, 10]

    # for alpha in alpha_list:
        
    #     #create model
    #     model = CollabFilterOneVectorPerItem(
    #     n_epochs=150, batch_size=32, step_size=0.2,
    #     n_factors=50, alpha= alpha)

    #     #initialize model with data
    #     model.init_parameter_dict(n_users, n_items, train_tuple)

    #     # Fit the model with SGD
    #     model.fit(train_tuple, valid_tuple)

    #     # Create a plot for model performance
    #     plt.figure(figsize=(8, 6))
    #     epoch_list = model.trace_epoch
    #     mae_list_train = model.trace_mae_train
    #     mae_list_valid = model.trace_mae_valid
    #     plt.plot(epoch_list, mae_list_train, label=f'Train - alpha = {alpha}')
    #     plt.plot(epoch_list, mae_list_valid, label=f'Valid - alpha = {alpha}')
    #     plt.title(f'MAE by epoch - alpha = {alpha}')
    #     plt.xlabel('Epoch')
    #     plt.ylabel('MAE')
    #     plt.legend()
    #     plt.savefig(f'alpha_{alpha}_graph.png')
    #     plt.close()

    #part 3
    import pandas as pd

    #get csv of selected movies
    selected_movies = pd.read_csv('../data_movie_lens_100k/select_movies.csv')
    
    #grab movie IDs
    selected_movie_ids = list(selected_movies['item_id'])

    # Create an instance of the model
    model = CollabFilterOneVectorPerItem(
        n_epochs=600, batch_size=32, step_size=0.2,
        n_factors=2, alpha=0.1)  # Choose appropriate parameters

    # Initialize the model's parameter dictionary
    model.init_parameter_dict(n_users, n_items, train_tuple)

    # Fit the model with the training data
    model.fit(train_tuple, valid_tuple)

    # Access the embedding matrices U and V from the param_dict attribute
    U_matrix = model.param_dict['U']  # User embedding matrix
    V_matrix = model.param_dict['V']  # Item embedding matrix

    #get only selected movies
    selected_V_matrix = V_matrix[selected_movie_ids]

    #plot embedding vectors
    plt.figure(figsize=(8, 6))
    plt.scatter(selected_V_matrix[:,0], selected_V_matrix[:,1])

    #plot movie titles on scatterplot
    for i, movie_id in enumerate(selected_movie_ids):
        movie_title = selected_movies[selected_movies['item_id'] == movie_id]['title'].values[0]
        plt.text(selected_V_matrix[i, 0], selected_V_matrix[i, 1], movie_title)



    plt.xlabel('Embedding Dimension 1')
    plt.ylabel('Embedding Dimension 2')
    plt.title('Embedding Vectors for Selected Movies')
    plt.savefig('embeddings.png')
    plt.close()





