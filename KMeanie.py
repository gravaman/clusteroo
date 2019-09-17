import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler


class KMeanie:
    def __init__(self, data_path, index=None, ycol=None):
        self.X = pd.read_csv(data_path)
        if index is not None:
            self.X = self.X.set_index(index)

        if ycol is not None:
            self.Y = self.X[ycol]
            self.X = self.X.drop(columns=[ycol])

        print(self.X.head())
        scaler = StandardScaler()
        normed = scaler.fit_transform(self.X.values)
        self.X_norm = pd.DataFrame(normed,
                                   columns=self.X.columns,
                                   index=self.X.index)

    def plot_distortion(self, low=1, high=7, tol=1e-4):
        distortion = []
        cluster_count = range(low, high)
        for i in cluster_count:
            kmeans = KMeans(n_clusters=i,
                            n_init=i,
                            init='random',
                            tol=tol,
                            random_state=170,
                            verbose=True).fit(self.X.values)
            distortion.append(kmeans.inertia_)

        plt.plot(cluster_count, distortion)
        plt.xlabel('# clusters')
        plt.ylabel('distortion')
        plt.show()

    def fit(self, n_clusters=2, n_init=2, tol=1e-4):
        self.model = KMeans(n_clusters=n_clusters,
                            n_init=n_init,
                            init='random',
                            tol=tol,
                            random_state=170,
                            verbose=True).fit(self.X_norm.values)
        cnt = range(n_clusters)
        ixs = [np.where(self.model.labels_ == i, True, False) for i in cnt]
        self.clusters = [self.X_norm[idx] for idx in ixs]
        for i, cluster in enumerate(self.clusters):
            coefs = np.corrcoef(cluster.values, rowvar=False)[-1, :-1]
            sort_coefs = np.argsort(coefs)[::-1][:]
            print(f'cluster {i} summary')
            print(f'count: {cluster.shape[0]}')
            print(f'feature correlations:')
            for coef in sort_coefs:
                print(self.X_norm.columns[coef], coefs[coef])
            print(cluster.head())

    def plot(self, yax='ev_sales', skey='pop_growth', anno_key=1):
        s = self.X_norm[skey].values*40
        plt.scatter(self.Y, self.X[yax], c=self.model.labels_, s=s)
        for idx in list(self.clusters[anno_key].index.values):
            plt.annotate(idx, (self.Y[idx], self.X.loc[idx, yax]),
                         textcoords='offset points', xytext=(0, 10))
        plt.xlabel('ev chg since 9/5')
        plt.ylabel('ev/sales')
        plt.show()


if __name__ == '__main__':
    meanie = KMeanie('data/enterprise_saas_v2.csv',
                     index='ticker',
                     ycol='ev_chg')
    meanie.fit(n_clusters=3, n_init=3)
    meanie.plot()
