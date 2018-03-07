import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import signal
from sklearn.preprocessing import RobustScaler
from sklearn.cluster import KMeans


class EcgMethods(object):

    def __init__(self):
        self.ts = None
    
    def get_segments(self, input_time_series, window_length, overlap=0.5):
        """
        Segments a time series with overlapping windows.
        
        -----
        
        input_time_series:
            an input time series
        
        returns:
            a list of numpy arrays
        """
        y = np.array(input_time_series)
        segments = []
        starts = [i for i in range(0, y.size, int(window_length*overlap))]
        for start in starts:
            seg = y[start: start+window_length]
            if len(seg) != window_length:  # this excludes too short at end
                continue
            segments.append(seg)
        
        return segments
    
    def get_cluster_components(self, series_segment):
        """
        Marks components of the ecg signal into 
        QRS and P and T waves

        -----

        series_segment:
            A time series segment of interest
        
        returns:
            The identity (e.g., R or S, Q, etc)
        """
        segment = np.array(series_segment)
        max_idx = segment.argmax()
        
        after_max_list = list(range(max_idx+1, len(segment)))
        if after_max_list:
            min_after_idx = after_max_list[np.argmin(segment[after_max_list])]
        else:
            min_after_idx = False
            
        before_max_list = list(range(max_idx))
        if before_max_list:
            min_before_idx = before_max_list[np.argmin(segment[before_max_list])]
        else:
            min_before_idx = False
            
        # P, T
        if segment.max() < 2:
            if (max_idx > len(segment)*0.5) & (max_idx < len(segment)*.75):
                return("pt_cluster")
        
        else:
            # Q
            if ((min_before_idx > len(segment)*0.1) &
                (min_before_idx < len(segment)*0.9) &
                (max_idx > len(segment)*0.8)):
                return("q_cluster")
            
            # S
            elif ((min_after_idx > len(segment)*0.1) &
                (min_after_idx < len(segment)*0.9) &
                (max_idx < len(segment)*0.2)):
                return("s_cluster")

            # R
            elif ((max_idx > len(segment)*0.1) &
                (max_idx < len(segment)*0.9)):
                return("r_cluster")

    def get_kmeans_centroids(self, time_series_segments, num_of_clusters,
            num_init, max_iter, random_state=0, printkmeans=True):
        self.kmeans = KMeans(n_clusters=num_of_clusters, random_state=random_state)
        self.kmeans.fit(full_segments)
        if printkmeans == True:
            print(self.kmeans)
    
    def get_cluster_indices(self):
        """
        Gets the indices for ecg wave components from the cluster centroids

        """
        for i, cl in enumerate(self.kmeans.cluster_centers_):
            clust_wave = get_cluster_components(cl)
            if clust_wave == 'pt_cluster':
                ptClusters.append(i)
            elif clust_wave == 'q_cluster':
                qClusters.append(i)
            elif clust_wave == 'r_cluster':
                rClusters.append(i)
            elif clust_wave == 's_cluster':
                sClusters.append(i)


class BandFilters(object):

    def __init__(self, input_series_to_filter):
        self.input_series_to_filter = input_series_to_filter

    def _butter_lowpass(self, highcut, fs, order=5):
        nyq = 0.5 * fs
        high = highcut / nyq
        b, a = signal.butter(order, [high], btype='low')
        return b, a
    
    def butter_lowpass_filter(self, input_series_to_filter,
                              highcut, fs, order=5):
        b, a = _butter_lowpass(highcut, fs, order=order)
        return signal.lfilter(b, a, input_series_to_filter)

    def _butter_bandpass(self, lowcut, highcut, fs, order=5):
        nyq = 0.5 * fs
        low = lowcut / nyq
        high = highcut / nyq
        b, a = signal.butter(order, [low, high], btype='band')
        return b, a

    def butter_bandpass_filter(self, input_series_to_filter,
                               lowcut, highcut, fs, order=5):
        b, a = _butter_bandpass(lowcut, highcut, fs, order=order)
        return signal.lfilter(b, a, input_series_to_filter)
    
    def smooth_series(self, input_series_to_filter, N):
        return np.convolve(input_series_to_filter,
                           np.ones((N,))/N, mode='same')
