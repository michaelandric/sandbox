import numpy as np


class TsAnalyzer(object):

    def __init__(self, input_time_series):
        """
        input_time_series:
            The input time series.
            Either a list of numbers or a numpy array.
        """
        self.y = np.array(input_time_series)

    def turnpoints(self):
        """
        ported the R version of this to python.
        https://cran.r-project.org/web/packages/pastecs/index.html
        
        x: 
            number list or 1-D numpy array
        
        return:
            dictionary with turnpoints parameters
            
        e.g. usage:
            tp_num_array = turnpoints(num_array)
            num_turnpoints = tp_num_array['nturns']
            peak_values = np.array(num_array)[tp_num_array['pos'][tp_num_array['peaks']]]
            pit_values = np.array(num_array)[tp_num_array['pos'][tp_num_array['pits']]]
        """
        from math import gamma
        
        assert len(set(self.y)) > 2, 'need at least 3 unique values'
        n = len(self.y)
        diffs = np.not_equal(np.hstack((self.y[0]-1, self.y[0:len(self.y)-1])),
                self.y)
        uniques = np.array(self.y)[diffs]
        n2 = len(uniques)
        poss = [i for i, a in enumerate(diffs) if a == True]
        exaequos = np.array((poss[1:n2]+[n])) - poss - 1
        
        m = n2 - 2
        reper = np.repeat([3, 2, 1], repeats=m)
        for i in range(int(len(reper)/m)):
            reper[i*m:i*m+m] += range(m)
        reper = reper - 1
        ex = np.array([np.array(self.y)[diffs][i] for i in reper]).reshape(int(len(reper)/m), m).T
        peaks = np.hstack((False, ex.max(1) == ex[:, 1], False))
        pits = np.hstack((False, ex.min(1) == ex[:, 1], False))
        tpts = peaks | pits
        if tpts.sum() == 0:
            nturns = 0
            firstispeak = False
            peaks = np.repeat(False, n2)
            pits = np.repeat(False, n2)
            tppos = np.nan
            proba = np.nan
            info = np.nan
        else:
            tppos = (poss + exaequos)[tpts]
            tptspos = np.arange(n2)[tpts]
            firstispeak = tptspos[0] == np.arange(n2)[peaks][0]
            nturns = len(tptspos)
            if (nturns < 2):
                inter = n2 + 1
                posinter1 = tptspos[0]
            else:
                inter = np.hstack((tptspos[1:nturns], n2-1)) - np.hstack((0, tptspos[:nturns-1])) + 1
                posinter1 = tptspos - np.hstack((0, tptspos[:(nturns - 1)]))
            posinter2 = inter - posinter1
            posinter = np.array((posinter1, posinter2)).max(0)
            if type(posinter) is np.ndarray:
                proba = 2/((inter * [gamma(i) for i in posinter]) * [gamma(i) for i in np.array((inter - posinter + 1))])
            if type(posinter) is np.int64:
                proba = 2/(inter * gamma(posinter) * gamma(inter - posinter + 1))
            info = -np.log2(proba)
        
        res_dict = {'data': self.y, 'points': uniques, 'pos': (poss+exaequos),
                    'exaequos': exaequos, 'nturns': nturns, 'firstispeak': firstispeak,
                    'peaks': peaks, 'pits': pits, 'tppos': tppos, 'proba': proba, 'info': info}

        return res_dict


    def amplitude_variance_asymmetry(self, s):
        """
        Amplitude Variance Asymmetry
        Requires turnpoints method
        self.y: 
            input vector
        s:
            smoothing window, e.g., 
            s = np.array((0.5, 1, 0.5))
            s = s/s.sum()
        
        return:
            the log ratio of variance in peaks to variance in pits
        """
        assert turnpoints, 'turnpoints method required!'
        
        xflt = np.convolve(self.y, s, mode='same')
        turnpoints_x = turnpoints(xflt)
        peak_positions = turnpoints_x['pos'][turnpoints_x['peaks']]
        pit_positions = turnpoints_x['pos'][turnpoints_x['pits']]
        assert peak_positions.size > 0, 'Only one or none peaks; cannot run AVA'
        assert pit_positions.size > 1, 'Only one or none pits; cannot run AVA'
        var_peaks = np.var(np.array(xflt)[peak_positions])
        var_pits = np.var(np.array(xflt)[pit_positions])
        logratio = np.log(var_peaks/var_pits)

        return logratio


    def magnitude_anomaly_finder(self, window_length, overlap,
                                 n_clusters, stddev_limit, show_test_segment=True):
        """
        uses numpy, pandas, sklearn
        y:
            number array
        window_length:
            length of the window to segment
        overlap:
            proportion of overlap between windows
        n_clusters:
            number of clusters to find
        stddev_limit:
            number of standard deviations to mark anomalous
        """
        import numpy as np
        import pandas as pd
        from sklearn.cluster import KMeans
        import matplotlib.pyplot as plt

        y = np.array(self.y, dtype=np.float)
        segments = []
        starts = [i for i in range(0, y.size, int(window_length*overlap))]
        for start in starts:
            seg = y[start: start+window_length]
            if len(seg) != window_length:  # this excludes too short at end
                continue
            segments.append(seg)


        full_segments = np.asarray(segments)
        windowed_segments = full_segments * np.hamming(window_length)
        
        # Find clusters:
        kmeans = KMeans(n_clusters=150, random_state=0).fit(windowed_segments)
        print(kmeans)
        
        # test segment and view it: 
        segmnt = np.random.randint((windowed_segments.shape[0]))
        print(segmnt)

        if show_test_segment == True:
            if plt:
                plt.plot(kmeans.cluster_centers_[kmeans.predict(
                    windowed_segments)[segmnt]], label='nearest centroid')
                plt.plot(windowed_segments[segmnt], label='windowed segment')
                plt.plot(full_segments[segmnt][:int(window_length/2)],
                         label='orig segment')
                plt.legend()
                plt.title('Test segment with centroid')
                plt.show()
            else:
                print("Won't plot test segment.")
                print("Need import matplotlib.pyplot as plt")
        
        # Reconstruct series from clusters
        slide_len = int(window_length/2)

        test_segments = []
        starts = [i for i in range(0, y.size, slide_len)]
        for start in starts:
            seg = y[start: start+window_length]
            if len(seg) != window_length:  # this excludes too short at end
                continue
            test_segments.append(seg)
        
        reconstruction = np.zeros(len(y))

        for segment_n, segment in enumerate(np.array(test_segments)):
            # don't modify data in segments
            segment = np.copy(segment)
            segment *= np.hamming(window_length)
            nearest_centroid_idx = kmeans.predict(segment.reshape(1, -1))[0]
            centroids = kmeans.cluster_centers_
            nearest_centroid = np.copy(centroids[nearest_centroid_idx])

            # overlay reconstructed segments with overlap of half a segment
            pos = segment_n * slide_len
            reconstruction[pos:pos+window_length] += nearest_centroid
        
        start_sample = int(y.shape[0] / 3)
        n_plot_samples = 1000

        error = (reconstruction[start_sample:start_sample+n_plot_samples] -
                 y[start_sample:start_sample+n_plot_samples])
        error_98th_percentile = np.percentile(error, 98)
        print('Maximum reconstruction error was {:.1f}'.format(error.max()))
        print('98th percentile of reconstruction error was {:.1f}'.format(
            error_98th_percentile))
        
        standardized = (error - error.mean()) / error.std()
        if plt:
            fig = plt.figure(figsize=(20, 10))
            fig.tight_layout()
            plt.plot(y[start_sample:start_sample+n_plot_samples],
                     label="Original signal", linewidth=3)
            plt.plot(reconstruction[start_sample:start_sample+n_plot_samples],
                     label="Reconstructed signal", color='y', linewidth=3)
            plt.plot(error, label="Reconstruction Error", color='r', linewidth=3)
            anomalous_stddev_limit = stddev_limit
            anomalous = np.where(np.abs(standardized) > anomalous_stddev_limit)[0]
            for idx in anomalous:
                plt.axvline(idx, color='m', alpha=0.2)
            plt.legend(fontsize=15)
            plt.title('Reconstructed and Anomalous signals')
            plt.show()
        else:
            print('Need import matplotlib.pyplot as plt')
