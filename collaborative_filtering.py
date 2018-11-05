import numpy as np
import pandas as pd
import scipy.sparse as sparse
from scipy.stats import percentileofscore
from time import time


# status generator
def range_with_status(total):
    """ iterate from 0 to total and show progress in console """
    n = 0
    while n < total:
        done = '#' * (n + 1)
        todo = '-' * (total - n - 1)
        s = '<{0}>'.format(done + todo)
        if not todo:
            s += '\n'
        if n > 0:
            s = '\r'+s
        print(s, end='', flush=True)
        yield n
        n += 1


class Base:
    """
    Basic stuff common in all classes
    """

    def __init__(self, num_factors=10, regularization=.015, sep=',', alpha=1, epsilon=1, num_iter=15,
                 number_of_recommendations=5,
                 float_type=np.float16, cold_train=True, learning_rate=0.05, reg_u=0.0025, reg_i=0.0025, reg_j=0.00025,
                 with_replacement=False,
                 multi_process_user=True, multi_process_item=False,
                 num_cores=None,
                 random_seed=None):
        """

        :param num_factors: int
            Number of factors using in Collaborative filtering

        :param regularization: float
            Regularization parameter

        :param sep: str
            Will feed to pandas.read_csv, sep parameter

        :param alpha: float
            Alpha parameter in ... function ..todo::

        :param num_iter: int
            Number of iterations fo training

        :param number_of_recommendations: int
            Number of items that are going to be recpmmended by predict

        :param float_type: type
            Type of float to be used

        :param cold_train: boolean
            If true the matrices will be reinitiated every time fit method is called

        :param multi_process_user: boolean
            If true, updating user matrix will be done in parallel

        :param multi_process_item: boolean
            If true, updating item matrix will be done in parallel

        :param random_seed: int
            Random seed
        """

        self.num_factors = num_factors
        self.cold_train = cold_train

        self.regularization = regularization
        self.reg_u = reg_u
        self.reg_i = reg_i
        self.reg_j = reg_j
        self.alpha = alpha
        self.epsilon = epsilon
        self.num_iter = num_iter
        self.learning_rate = learning_rate
        self.with_replacement = with_replacement
        # self.user_col = 'UserID'
        # self.item_col = 'ItemID'
        self.rating_col = None
        self.sep = sep
        self.y_matrix = None
        self.x_matrix = None
        self._float_type = float_type

        self.number_of_recommendations = number_of_recommendations

        self.multi_process_user = multi_process_user
        self.multi_process_item = multi_process_item
        self.num_cores = num_cores
        self.random_seed = random_seed
        self.train_dic = None
        self._p_matrix = None

    _DEBUG = True

    def printd(self, *args):
        if self._DEBUG:
            print(*args, flush=True)

    def load(self, df, rating_col=None, bprmf=False, **args):
        """
        Take input in form of UserID, ItemID and convert it to userID, ItemID number of instances

        :param rating_col:
        :param df:
        :param user_col:
        :param item_col:
        :param args:
        :return:
        """
        if isinstance(df, str):
            df = pd.read_csv(df, header=-1, dtype='object', **args)

        self.users = list(df.iloc[:, 0].unique())
        self.items = list(df.iloc[:, 1].unique())

        # will be used for omitting repeated items
        f = df.groupby(list(df.columns.values[0:2])).agg({df.columns[0]: 'count'})
        self.train_dic = {self._get_index(self.users, [k])[0]:
                              list(self._get_index(self.items, f.loc[k].index)) for k in f.index.levels[0]}

        # will be used in choosing a random item with no interaction with the chosen user
        if bprmf:
            g = df.groupby(list(df.columns.values)[0:2]).agg({df.columns[1]: 'count'})
            self.item_dic = {self._get_index(self.items, [k])[0]:
                                 set(self._get_index(self.users, g.xs(k, level=1).index)) for k in g.index.levels[1]}

        if rating_col is None:
            rating_col = 'rating'
            df[rating_col] = 1

        print('There are %i users and %i items in the trainig dataset' % (len(self.users), len(self.items)), flush=True)

        user_index = self._get_index(self.users, df.iloc[:, 0])#[self.users.index(u) for u in df[user_col]]
        item_index = self._get_index(self.items, df.iloc[:, 1])#[self.items.index(i) for i in df[item_col]]

        # interactions matrix
        int_matrix = sparse.coo_matrix((df[rating_col], (user_index, item_index)),
                                       shape=(len(self.users), len(self.items)), dtype=self._float_type)

        return int_matrix

    def _remove_repeated(self, test_dic):
        keys = list(test_dic.keys())
        for key in keys:
            if key in set(self.users):
                key_index = self._get_index(self.users, [key])[0]
            else:
                _ = test_dic.pop(key)
                continue

            if isinstance(test_dic[key], list) or isinstance(test_dic[key], set):
                value_test = test_dic[key]
            else:
                value_test = [test_dic[key]]

            if set(value_test).issubset(set(self.items)):
                if set(self._get_index(self.items, [test_dic[key]])).issubset(set(self.train_dic[key_index])):
                    _ = test_dic.pop(key)

        return test_dic

    def eval_1(self, pred_result, df, metric, repeated_items=False, **args):
        """
        assuming df contains user_id and a single recommendation for it
        
        :param metric: str
            The metric to be used from ['percentile', 'hit_rate']
         
        :param pred_result: dict
            prediction results for users in test dataset
            
        :param df: pandas.DataFrame or string
            userID, Recommended itemID. Or path to a csv file having these
            
        :return: 
            Sum of percentile scores of expected recommendation in predicted recommendations divided by size of test 
            data set. Or hit rate
        """
        assert metric in {'percentile', 'hit_rate'}, "metric have to be in ['percentile', 'hit_rate']"

        if isinstance(df, str):
            test = pd.read_csv(df, header=-1, dtype='object', **args)
        else:
            test = df.copy()
        if isinstance(test, pd.DataFrame):
            test_dic = dict(zip(test.iloc[:, 0], test.iloc[:, 1]))
        else:
            test_dic = test

        if not repeated_items:
            test_dic = self._remove_repeated(test_dic)

        score = []
        exp = 0
        for key in test_dic.keys():
            try:
                if metric == 'percentile':
                    score.append(percentileofscore(list(pred_result[key].values()), pred_result[key][test_dic[key]]))
                elif metric == 'hit_rate':
                    if isinstance(test_dic[key], list) or isinstance(test_dic[key], set):
                        value_test = test_dic[key]
                    else:
                        value_test = [test_dic[key]]
                    score.append(int(set(value_test).intersection(set(pred_result[key].keys())) != set() ))
            except:
                exp += 1
                continue
        # print(exp, ' exceptions happed in eval by %s' %metric)
        return sum(score) / len(test_dic)

    def _get_index(self, source, target):
        """
        Output the indices of source where the elements of target happen
        :param source: list
        :param target: 
        :return: 
        list the same size as target
        """
        return [source.index(u) for u in target]

    def _load_to_predict(self, df, **args):
        """
        Indices of input for predict in users
        :param df2:
        :param args: 
        :return: 
        """
        if isinstance(df, str):
            df2 = pd.read_csv(df, header=-1, dtype='object', **args)
            df2 = df2.iloc[:, 0]
        else:
            df2 = df.iloc[:, 0]

        df2 = df2[df2.isin(self.users)]
        if self._DEBUG:
            self.predict_items= df2

        # TODO redundant having the above line
        assert df2.isin(self.users).all(), 'There are users not seen on train'

        return self._get_index(self.users, df2.values)

    def _setxy(self, sparse_output=True):
        if self.y_matrix is None or self.cold_train:
            np.random.seed(self.random_seed)
            randoms = np.random.RandomState(self.random_seed)
            self.y_matrix = randoms.normal(0, .01, (len(self.items), self.num_factors))
            if sparse_output:
                self.y_matrix = sparse.csr_matrix(self.y_matrix, dtype=self._float_type)

        if self.x_matrix is None or self.cold_train:
            self.x_matrix = randoms.normal(0, .01, (len(self.users), self.num_factors))

            if sparse_output:
                self.x_matrix = sparse.csr_matrix(self.x_matrix, dtype=self._float_type)

    def predict(self, list_of_users, filename=None, print_results=False, repeated_items=False, **args):
        """
        Predict the recommendations.

        :param list_of_users: list like
            List of users for recommending. They have to be a subset of users in train data

        :param filename: str optional
            Path to write the prediction results

        :param print_results: boolean
            If true, the results will be printed on the screen

        :param repeated_items: boolean
            If True, the used items will be recommended again to the user

        :param args: Will be feed to _load to predict
        :return: dic
            keys: list_of_users
            values: dictionary of recommendations where keys are list of items and values the corresponding score.

        """
        t0 = time()

        user_list = self._load_to_predict(list_of_users, **args)

        if isinstance(self.x_matrix, (sparse.coo.coo_matrix, sparse.csr.csr_matrix, sparse.csc.csc_matrix)):
            self.x_matrix = self.x_matrix.toarray()
            self.y_matrix = self.y_matrix.toarray()

        _y_predicted = self.x_matrix[user_list, :].dot(self.y_matrix.T)
        items = np.array(self.items)

        prediction_result = dict()
        to_dump = []

        # FIXME this for loop is parallelizable
        for i, u in enumerate(user_list):
            rates = _y_predicted[i, :]
            if not repeated_items:
                rates[self.train_dic[u]] = 0

            s = rates.argsort()
            s = s[::-1][:self.number_of_recommendations]

            keys = items[s]
            values = rates[s]
            suggestions_for_u = dict(zip(keys, values))
            prediction_result[self.users[u]] = suggestions_for_u

            if print_results:
                print('%s\t' % self.users[u], ['%s:%s' % (a, b) for a, b in zip(keys, values)], flush=True)

            #             Investigate the speed of this loop first, if worrying about this is needed.
            if filename is not None:
                to_dump.append(
                    ("%s\t[%s]" % (self.users[u], ','.join(['%s:%s' % (a, b) for a, b in zip(keys, values)]))))

        if filename is not None:
            self._dump_string_list(to_dump, filename)

        print('predict in %i seconds' %(time() - t0))


        return prediction_result


class BPRMF(Base):
    """
    Class for traing matrix factorization with Bayesian Personalized Ranking (BPR) method.

    - Steffen Rendle , Christoph Freudenthaler , Zeno Gantner , Lars Schmidt-Thieme, BPR: Bayesian personalized ranking
      from implicit feedback, Proceedings of the Twenty-Fifth Conference on Uncertainty in Artificial Intelligence,
      p.452-461, June 18-21, 2009, Montreal, Quebec, Canada
    """

    def _sigmoid_d(self, x):
        return np.exp(-x) / (1 + np.exp(-x))

    def _choose_u_i(self, int_matrix):
        nonzeros = int_matrix.nonzero()
        indices = np.random.choice(len(nonzeros[0]), len(nonzeros[0]), replace=self.with_replacement)
        return [(int_matrix.nonzero()[0][i], int_matrix.nonzero()[1][i]) for i in indices]

    def _random_item(self, u):
        j = np.random.choice(len(self.items))
        while u in self.item_dic[j]:
            j = np.random.choice(len(self.items))

        return j

    def _row(self, matrix, row, rownum):
        """
        Return a matrix with same shape as matrix, row in its row number row and 0 elsewhere
        """

        m = np.zeros(matrix.shape)
        m[rownum, :] = row
        return m

    def _hat(self, u, i, j):
        xhat = (self.x_matrix[u, :] * self.y_matrix[i, :]).sum() - \
               (self.x_matrix[u, :] * self.y_matrix[j, :]).sum()
        return xhat

    def _update(self, u, i, j):

        xhat = self._hat(u, i, j)

        x_update = self.learning_rate * (self._sigmoid_d(xhat) * (self.y_matrix[i, :] - self.y_matrix[j, :]) +
                                         self.reg_u * self.x_matrix[u, :])
        y_update_i = self.learning_rate * (self._sigmoid_d(xhat) * self.x_matrix[u, :] +
                                           self.reg_i * self.y_matrix[i, :])
        y_update_j = self.learning_rate * (-self._sigmoid_d(xhat) * self.x_matrix[u, :] +
                                           self.reg_j * self.y_matrix[j, :])

        self.x_matrix[u, :] = self.x_matrix[u, :] + x_update
        self.y_matrix[i, :] = self.y_matrix[i, :] + y_update_i
        self.y_matrix[j, :] = self.y_matrix[j, :] + y_update_j

    def fit(self, df):
        """
        Train the model using BPR

        :param df: see load
        """
        int_matrix = \
            self.load(df=df, rating_col=self.rating_col, sep=self.sep, bprmf=True)

        if self.x_matrix is None or self.cold_train:
            np.random.sample(self.random_seed)

        self._setxy(sparse_output=False)

        t0 = time()
        for ITER in range(self.num_iter):
            ui_list = self._choose_u_i(int_matrix)

            for i_ in range(len(int_matrix.nonzero()[0])):
                u, i = ui_list[i_]
                j = self._random_item(u)
                self._update(u, i, j)

            print('', end='\r', flush=True)
            print('iteration %i of %i, %i seconds'
                  % ((ITER + 1), self.num_iter, time() - t0), end='', flush=True)

        print('', flush=True)