from asyncore import loop
import pandas as pd
import numpy as np
from tqdm import tqdm

class Kmeans():
    def __init__(self,
                 mode='mean'):
        '''
        K means clustering algorithm

        Parameters:
        - mode (str): The mode used to calculate the centroid. Default is 'mean'.
        '''
        self.mode=mode
        self.last_centriods = None

    def fit(self,
            df=None,
            k=None,
            start_centriods=None,
            inplace=False,
            max_iter=1000,
            explain=False):
        '''
        fit K means with `k` group

        Parameters:
        - df (pandas.DataFrame): The input dataframe containing the data points.
        - k (int): The number of clusters.
        - start_centriods (numpy.ndarray): The initial centroids. If None, random data points will be selected as centroids.
        - inplace (bool): Whether to modify the input dataframe in place. Default is False.
        - max_iter (int): The maximum number of iterations. Default is 1000.
        - explain (bool): Whether to print the intermediate steps for explanation. Default is False.

        Returns:
        - pandas.DataFrame: The input dataframe with an additional 'label' column indicating the cluster label for each data point.
        '''
        if not inplace:
            df = df.copy()
        df['label'] = None

        # initialization
        centriods = None
        if start_centriods is None:
            centriods = df.sample(k).iloc[:, :-1].to_numpy().astype(float)
        elif len(start_centriods) != k:
            raise ValueError('Invalid centriod\'s shape')
        else:
            centriods = start_centriods

        if explain:
            print(f"Init centriod : {centriods}")

        # loop until done
        looper = tqdm(range(max_iter), desc=f'Fitting K means for k={k}') \
            if not explain else range(max_iter)
        for iter in looper:
            # assign
            if explain:
                print(f"Assign #{iter+1}")
            for row_idx in range(df.shape[0]):
                data_point = df.iloc[row_idx, :-1].to_numpy()
                min_dist = float('inf')
                centriod_idx = 0
                for idx, centriod in enumerate(centriods):
                    dist = self.distance(centriod, data_point)
                    if dist < min_dist:
                        min_dist = dist
                        centriod_idx = idx
                if df.iloc[row_idx, -1] != centriod_idx and explain:
                    print(f"row #{row_idx} is change from group {df.iloc[row_idx, -1]} to {centriod_idx}")
                df.iloc[row_idx, -1] = centriod_idx
            if explain:
                print(df)

            # update
            change = False
            if explain:
                print(f"Update #{iter+1}")
            for centriod_idx in range(k):
                new_centriod = self.mean_point(df[df['label']==centriod_idx])
                dist = self.distance(centriods[centriod_idx], new_centriod)
                if not explain:
                    looper.set_postfix({'centriod_move':dist})
                if dist > 1e-5:
                    change = True
                if explain:
                    print(f"for centriod #{centriod_idx} change from {centriods[centriod_idx]} to {new_centriod} which diff {dist} units")
                centriods[centriod_idx] = new_centriod
            if explain:
                print(df)

            # break condition
            if not change:
                if explain:
                    print(f"Done at #{iter+1}")
                else:
                    looper.set_description_str(f'Fitting K means for k={k} done at #{iter+1}')
                break
        self.last_centriods = centriods
        return df

    def mean_point(self, df):
        '''
        calculate centriod of given data point

        Parameters:
        - df (pandas.DataFrame): The input dataframe containing the data points.

        Returns:
        - numpy.ndarray: The centroid of the given data points.
        '''
        df = df.copy()
        data_point = df.iloc[:, :-1].to_numpy()
        mean = data_point.mean(axis=0)
        return mean


    def distance(self, x1, x2):
        '''
        calculate by Euclidean distance

        Parameters:
        - x1 (numpy.ndarray): The first data point.
        - x2 (numpy.ndarray): The second data point.

        Returns:
        - float: The Euclidean distance between x1 and x2.
        '''
        return np.sqrt(((x1-x2)**2).sum())
