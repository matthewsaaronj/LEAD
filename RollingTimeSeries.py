import numpy as np
import pandas as pd


class RollingTimeSeriesCV:
    '''
    Expects X as a dataframe, with date column
    '''
    
    def __init__(self, n_splits=3, test_percent = 0.10):
        self.n_splits = n_splits
        self.test_percent = test_percent
        self.train_size = ''
        self.test_size = ''
        self.test_start = ''
        
    def split(self, X, y, date_column):
        # Define variables for identifying rolling windows
        time_values = X[date_column].unique()
        self.test_size = int(time_values.shape[0] * self.test_percent)
        required_test_observations = self.test_size * self.n_splits
        self.test_start = time_values[~required_test_observations]
        self.train_size = time_values[time_values < self.test_start].shape[0]
        initial_test_loc = np.where(time_values == self.test_start)[0][0]
        
        # Build CV Splits
        cv_splits = list()
        
        for x in np.arange(0, self.n_splits):
            rolling_x = (x * self.test_size)
            test_position = initial_test_loc + rolling_x
            train_indices = X.index[X[date_column].isin(time_values[(0 + rolling_x):(self.train_size + rolling_x)])].values
            test_indices = X.index[X[date_column].isin(time_values[test_position:(test_position + self.test_size)])].values
            
            cv_splits.append((train_indices, test_indices))
            
        return cv_splits