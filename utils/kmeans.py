# !/usr/bin/python3
# coding: utf-8
# @Author: lixiang
# @Date: 2023-02-17

import numpy as np

def normalize(x, axis=-1, p=2):
    lp_norm = np.atleast_1d(np.linalg.norm(x, p, axis))
    lp_norm[lp_norm == 0] = 1
    
    return x / np.expand_dims(lp_norm, axis)

def euclidean_distance(one_sample, X):
    one_sample = one_sample.reshape(1, -1)
    X = X.reshape(X.shape[0], -1)
    distances = np.power(np.tile(one_sample, (X.shape[0], 1)) - X, 2).sum(axis=1)
    
    return distances

class Kmeans():
    """Kmeans 聚类算法.
    
    Parmaeters:
    ---------------------------
    K:              int, 聚类数
    max_iterations: int, 最大迭代次数
    varpsilon:      float, 判断是否收敛，即误差小于varpsilon
    
    """
    def __init__(self, k=2, max_iterations=500, varepsilon=0.0001):
        self.k = k
        self.max_iterations = max_iterations
        self.varepsilon = varepsilon
        
    ## 随机选取k个样本作为初始的聚类中心
    def init_random_centroids(self, X):
        n_samples, n_features = np.shape(X)
        centroids = np.zeros((self.k, n_features))
        for i in range(self.k):
            centroid = X[np.random.choice(range(n_samples))]
            centroids[i] = centroid
        
        return centroids
    
    ## 返回距离该样本最近的一个中心索引
    def _closest_centroid(self, sample, centroids):
        distances = euclidean_distance(sample, centroids)
        closest_i = np.argmin(distances)
        
        return closest_i
    
    ## 将所有样本进行归类，归类规则就是将该样本归类到其最近的中心
    def create_clusters(self, centroids, X):
        n_samples = np.shape(X)[0]
        clusters = [[] for _ in range(self.k)]
        for sample_i, sample in enumerate(X):
            centroid_i = self._closest_centroid(sample, centroids)
            clusters[centroid_i].append(sample_i)
        
        return clusters
    
    ## 更新中心
    def update_centroids(self, clusters, X):
        n_features = np.shape(X)[1]
        centroids = np.zeros((self.k, n_features))
        for i, cluster in enumerate(clusters):
            centroid = np.mean(X[cluster], axis=0)
            centroids[i] = centroid
        
        return centroids
    
    ## 将所有样本进行归类，其所在类别索引即为该类别标签
    def get_cluster_labels(self, clusters, X):
        y_pred = np.zeros(np.shape(X)[0])
        for cluster_i, cluster in enumerate(clusters):
            for sample_i in cluster:
                y_pred[sample_i] = cluster_i
        
        return y_pred
    
    ## 对整个数据集X进行Kmeans聚类，返回聚类标签
    def predict(self, X):
        centroids = self.init_random_centroids(X)
        ## 迭代，直到算法收敛
        for _ in range(self.max_iterations):
            clusters = self.create_clusters(centroids, X)
            former_centroids = centroids
            
            # 计算新的聚类中心
            centroids = self.update_centroids(clusters, X)
            
            diff = centroids - former_centroids
            if diff.any() < self.varepsilon:
                break
        
        return centroids, self.get_cluster_labels(clusters, X)
