from sklearn.metrics.pairwise import pairwise_distances
from sklearn.metrics import mean_squared_error
from heapq import nlargest
from tqdm import tqdm
from math import sqrt
import pandas as pd
import numpy as np
import os.path


def load_data(data_dir, filename):
    """
    Load CSV data to DataFrame

    Args:
        data_dir: Data directory
        filename: Filename

    Returns:
        A DataFrame with columns of 'UserID', 'ItemID', 'Rating', 'Timestamp'
    """

    fields = ['UserID', 'ItemID', 'Rating', 'Timestamp']

    return pd.read_csv(os.path.join(data_dir, filename), sep='\t', names=fields)


def get_matrix(df, num_users, num_items, col='Rating'):
    """
    Compose a rating matrix from DataFrame whose rows are (UserID-1),columns are (ItemID-1),
    and the rating is the value at this row and column. Any unobserved ratings are zero.

    Args:
        df: DataFrame, columns=['UserID', 'ItemID', 'Rating', ...]
        num_users: Number of users
        num_items: Number of items
        col:       Column name (default: 'Rating')
            
    Returns:
        matrix: 2D NumPy array.
    """

    matrix = np.zeros((num_users, num_items))

    # Populate the matrix
    for user_id, item_id, value in df[['UserID', 'ItemID', col]].itertuples(index=False):
        matrix[user_id - 1, item_id - 1] = value

    return matrix


class Baseline(object):
    """
    Baseline recommendation system based on users' average ratings or items' popularity

    Args:
        num_users: Number of users.
        num_items: Number of items.
        method_name: Filtering method, e.g., 'user_average' or 'popularity'.
        processor: Processing function handler.
    """

    def __init__(self, num_users, num_items, method_name, processor=get_matrix):
        self.num_users = num_users
        self.num_items = num_items
        self.method_name = method_name
        self.method = self._get_method(self.method_name)
        self.processor = processor
        self.pred_col_name = self.method_name

        self._model = None
        
    def _get_method(self, method_name):
        """Method switcher"""
        switcher = {'popularity': self.popularity,
                    'user_average': self.user_average}
        
        return switcher[method_name]

    def user_average(self, train_matrix):
        """
        Method based on users' average rating

        Args:
            train_matrix: 2D NumPy array of training data.

        Returns:
            pred_matrix: 2D NumPy array of the same dimensions and row/column IDs
                         as train_matrix, but anywhere there is a 0 in train_matrix,
                         there should be a predicted value in pred_matrix.
        """
        pred_matrix = np.zeros((self.num_users, self.num_items))

        for (user, item), rating in np.ndenumerate(train_matrix):
            # Extract the items the user already rated
            user_vector = train_matrix[user, :]
            rated_items = user_vector[user_vector.nonzero()]

            # If not empty, calculate average and set as rating for the current item
            item_avg = rated_items.mean() if rated_items.size else 0
            pred_matrix[user, item] = item_avg

        return pred_matrix

    def popularity(self, train_matrix):
        """
        Method based on items' popularity

        Args:
            train_matrix: 2D NumPy array of training data.

        Returns:
            pred_matrix: 2D NumPy array of the same dimensions and row/column IDs
                         as train_matrix, but anywhere there is a 0 in train_matrix,
                         there should be a predicted value in pred_matrix.
        """
        pred_matrix = np.zeros((self.num_users, self.num_items))

        # Define function for converting 1-5 rating to 0/1 (like / don't like)
        vf = np.vectorize(lambda x: 1 if x >= 4 else 0)

        # For every item calculate the number of people liked (4-5) divided by the number of people that rated
        item_pop = np.zeros(self.num_items)
        for item in range(self.num_items):
            num_users_rated = len(train_matrix[:, item].nonzero()[0])
            num_users_liked = len(vf(train_matrix[:, item]).nonzero()[0])
            
            item_pop[item] = num_users_liked / num_users_rated if num_users_rated else 0
            
        for (user, item), rating in np.ndenumerate(train_matrix):
            pred_matrix[user, item] = item_pop[item]

        return pred_matrix

    def fit(self, train_df):
        """
        Fit the model with training data with the specified method and
        return predictions to the model. Model stores the predicted UI matrix.

        Args:
            train_df: A DataFrame to be trained.
        """

        train_matrix = self.processor(train_df, self.num_users, self.num_items)
        self._model = self.method(train_matrix)

    def predict(self, test_df, copy=False):
        """
        Return a DataFrame (copy) with predictions in the test DataFrame

        Args:
            test_df: A DataFrame to be tested
            copy:    Copy flag

        Returns:
            pred_df: A DataFrame with predictions
        """

        pred_df = test_df.copy() if copy else test_df
        pred_df[self.pred_col_name] = np.nan
        
        for index, user_id, item_id in tqdm(pred_df[['UserID', 'ItemID']].itertuples()):
            pred_df.ix[index, self.pred_col_name] = self._model[user_id - 1, item_id - 1]

        return pred_df
        
    def get_model(self):
        """Return predicted UI matrix"""

        return self._model
    
    def get_pred_col(self):
        """Return prediction column name"""

        return self.pred_col_name
    
    def reset(self):
        """Reuse the instance of the class by removing model"""
        self._model = None


class CollaborativeFiltering(Baseline):
    """
    Collaborative filtering recommendation system based on similarity (e.g., user-user, item-tem)
    methods (e.g., cosine, Euclidean)

    Baseline's function handler 'method' will be overwritten by the new input 'method'

    Args:
        base:        Filtering base from ['user', 'item']. User- or item-based similarity
        method_name: Similarity method from ['cosine', 'euclidean']
        processor:   Processing function handler
    """
    def __init__(self, num_users, num_items, base, method_name, processor=get_matrix):
        super(CollaborativeFiltering, self).__init__(num_users,
                                                     num_items,
                                                     method_name,
                                                     processor=processor)
        self.base = base
        # Method overwrites parent method
        self.method_name = method_name
        self.method = self._get_method(self.method_name)
        # this overwrites as well
        self.pred_col_name = self.base + '-' + self.method_name
    
    def _get_method(self, method_name):
        """
        Method switcher (overwrites parent method)

        Args:
            method_name: Method name from 'cosine' or 'euclidean'
        """
        switcher = {'cosine': self.cosine,
                    'euclidean': self.euclidean}
        
        return switcher[method_name]
    
    @staticmethod
    def cosine(matrix):
        """
        Cosine similarity

        Args:
            matrix: same as the rating matrix with R rows and C columns.
                    Outputs an R x R similarity_matrix S where each S_ij
                    should be the cosine similarity between row i and
                    row j of matrix.
        Returns:
            Cosine similarity matrix
        """
        return 1 - pairwise_distances(matrix, metric='cosine')
    
    @staticmethod
    def euclidean(matrix):
        """
        Euclidean similarity
        
        Args:
            matrix: same as the rating matrix with R rows and C columns.
                    Outputs an R x R similarity_matrix S where each S_ij
                    should be the Euclidean similarity between row i and
                    row j of matrix.

        Returns:
            Euclidean similarity matrix
        """
        return 1 / (1 + pairwise_distances(matrix, metric='euclidean'))
        
    def fit(self, train_df):
        """
        Fit the model with training data (DataFrame) with the specified method and
        return predicted matrix to the model.

        This method overwrites parent method

        Args:
            train_df: A DataFrame to be trained.
        """

        train_matrix = self.processor(train_df, self.num_users, self.num_items)
        # Initialize the predicted rating matrix with zeros
        norm_matrix = np.zeros(train_matrix.shape)
        norm_matrix[train_matrix.nonzero()] = 1

        if self.base == 'user':
            uu_similarity = self.cosine(train_matrix)
            normalizer = np.matmul(uu_similarity, norm_matrix)
            normalizer[normalizer == 0] = 1e-5
            
            pred_matrix = np.matmul(uu_similarity, train_matrix) / normalizer

        elif self.base == 'item':
            ii_similarity = self.cosine(train_matrix.T)
            normalizer = np.matmul(norm_matrix, ii_similarity)
            normalizer[normalizer == 0] = 1e-5

            pred_matrix = np.matmul(train_matrix, ii_similarity) / normalizer
        else:
            raise Exception('No other option available')

        user_avg = np.sum(train_matrix, axis=1) / np.sum(norm_matrix, axis=1)
        cols = np.sum(pred_matrix, axis=0)
        pred_matrix[:, cols == 0] = pred_matrix[:, cols == 0] + np.expand_dims(user_avg, axis=1)

        self._model = pred_matrix


class CrossValidation(object):
    """
    Cross validation

    Args:
        num_users:
        num_items:
        metric_name:
        data_dir:
        processor:
    """
    def __init__(self, num_users, num_items, metric_name, data_dir, processor=get_matrix):
        self.num_users = num_users
        self.num_items = num_items
        self.folds = self._get_folds(data_dir)
        self.metric_name = metric_name
        self.metric = self._get_metric(self.metric_name)
        self.processor = processor
        
    def _get_metric(self, metric_name):
        """Metric switcher"""
        switcher = {'RMSE': self.rmse,
                    'PatK': self.patk,
                    'RatK': self.ratk}
        
        return switcher[metric_name]
    
    @staticmethod
    def rmse(pred_df, pred_col, true_col='Rating'):
        """
        RMSE metric

        Args:
            pred_df:  DataFrame with predictions.
            pred_col: Column corresponding to the prediction
            true_col: Column corresponding to the true rating
        """

        return sqrt(mean_squared_error(pred_df[pred_col], pred_df[true_col]))

    def patk(self, pred_df, k, pred_col, true_col='Rating'):
        """
        Precision at k, P@k

        Args:
            pred_df:  DataFrame with predictions.
            k:        Top-k items relevant
            pred_col: Column corresponding to the prediction
            true_col: Column corresponding to the true rating

        Returns:
            Average P@k
        """

        pred_matrix = self.processor(pred_df, self.num_users, self.num_items, pred_col)
        test_matrix = self.processor(pred_df, self.num_users, self.num_items, true_col)

        precisions = []

        # Define function for converting 1-5 rating to 0/1 (like / don't like)
        vf = np.vectorize(lambda x: 1 if x >= 4 else 0)

        for user_id in range(self.num_users):
            # Pick top K based on predicted rating
            user_pred = pred_matrix[user_id, :]
            topk = nlargest(k, range(len(user_pred)), user_pred.take)

            # Convert test set ratings to like / don't like
            user_test = vf(test_matrix[user_id, :]).nonzero()[0]

            precisions.append(sum(1 for item in topk if item in user_test) / len(topk))

        return np.mean(precisions)

    def ratk(self, pred_df, k, pred_col, true_col='Rating'):
        """
        Recall at k, R@k

        Args:
            pred_df:  DataFrame with predictions.
            k:        Top-k items retrieved
            pred_col: Column corresponding to the prediction
            true_col: Column corresponding to the true rating

        Returns:
            Average R@k
        """

        pred_matrix = self.processor(pred_df, self.num_users, self.num_items, pred_col)
        test_matrix = self.processor(pred_df, self.num_users, self.num_items, true_col)

        recalls = []

        # Define function for converting 1-5 rating to 0/1 (like / don't like)
        vf = np.vectorize(lambda x: 1 if x >= 4 else 0)

        for user_id in range(self.num_users):
            # Pick top K based on predicted rating
            user_pred = pred_matrix[user_id, :]
            topk = nlargest(k, range(len(user_pred)), user_pred.take)
            user_test = vf(test_matrix[user_id, :]).nonzero()[0]

            # Ignore user if has no ratings in the test set
            if len(user_test) == 0:
                continue

            recalls.append(sum(1 for item in topk if item in user_test) / len(user_test))

        return np.mean(recalls)
    
    @staticmethod
    def _get_folds(data_dir):
        """
        """
        folds = []
        data_types = ['u{}.base', 'u{}.test']
        for i in range(1, 6):
            train_set = load_data(data_dir, data_types[0].format(i))
            test_set = load_data(data_dir, data_types[1].format(i))
            folds.append([train_set, test_set])
        return folds
    
    def run(self, algorithms, k=1):
        """
        cross-validation

        Args:
            algorithms: A list of algorithms. e.g. [user_cosine_recsys, item_euclidean_recsys]
            k: Top-k for P@k and R@k
        """

        scores = {}
        for algo in algorithms:
            pred_col = algo.get_pred_col()
            fold_scores = []
            for fold in self.folds:
                algo.reset()
                # Train
                algo.fit(fold[0], self.num_users, self.num_items)
                # Predict
                pred = algo.predict(fold[1])
                fold_scores.append(self.metric(pred, k, self.num_users, self.num_items, pred_col))
            scores[pred_col] = fold_scores
    
        return scores
