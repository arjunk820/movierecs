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
            U=0.01 * random_state.randn(n_users, self.n_factors), # User hidden
            V=0.01 * random_state.randn(n_items, self.n_factors), # Item hidden
            )


    def predict(self, user_id_N, item_id_N,
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
        mse = ag_np.mean((y_true - y_pred) ** 2)

        regularization_term = 0
        for key in param_dict:
            regularization_term += ag_np.sum(param_dict[key] ** 2)

        # Total loss: MSE + alpha * regularization
        total_loss = mse + self.alpha * regularization_term

        return total_loss 


if __name__ == '__main__':

    # Load the dataset
    train_tuple, valid_tuple, test_tuple, n_users, n_items = \
        load_train_valid_test_datasets()
    # Create the model and initialize its parameters
    # to have right scale as the dataset (right num users and items)
    model = CollabFilterOneVectorPerItem(
        n_epochs=600, batch_size=32, step_size=0.2,
        n_factors=10, alpha=0.0)
    model.init_parameter_dict(n_users, n_items, train_tuple)

    # Fit the model with SGD
    epochs, trainMAE, validMAE = model.fit(train_tuple, valid_tuple)
    plt.plot(epochs, trainMAE, validMAE)
    plt.title('MAE by epoch - K=2')
    plt.xlabel('Epoch')
    plt.ylabel('MAE')
    plt.legend()
    plt.ylim(0.5, 1.5)
    plt.savefig('k2_graph')