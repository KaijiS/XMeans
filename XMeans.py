# -*- coding: utf-8 -*-
import numpy as np
from scipy import stats 
from sklearn.cluster import KMeans
from sklearn.neighbors import KNeighborsClassifier

"""-----------------------------------------------------------------------------

使用方法
import numpy as np 
from XMeans import XMeans

vectors = np.array([[  3,  3],
                    [  3,  0],
                    [ 17, 14],
                    [ 18, 15],
                    [  4,  1],
                    [ 16, 16],
                    [  4,  2],
                    [ 15, 15]])

# vectorsをクラスタリングしてラベルを付与
cluster_label = XMeans().fit_predict(vectors)
print(cluster_label)
> [1. 1. 0. 0. 1. 0. 1. 0.]

# vectorsをクラスタリング(学習)
xmeans = XMeans().fit(vectors)
# クラスタリング後の各クラスタの重心を利用してラベル付与(学習では使用していない未知のデータでも可)
print(xmeans.predict(vectors))
> [1. 1. 0. 0. 1. 0. 1. 0.]
# xmeansクラスタリング法のMSEに基づくスコアを算出)
print(xmeans.score(vectors))
> 2.5476205269238843


共分散も計算するか否かも引数(covariance)で指定できる　指定しなければデフォルトでは計算しない
参考論文では共分散を計算しなくとも同程度の結果になったらしい
共分散も計算しちゃうとランク落ちしがちだからオススメはしない
でもでも、データによっては共分散を計算しなくともランク落ちしちゃうから、その時はどんまい.ランク落ち時の対策法はネットにあるので調べましょう。
(例えば似ているデータが複数ある時や次元が高すぎる時)


参考論文:	http://www.rd.dnc.ac.jp/~tunenori/doc/xmeans_euc.pdf
参考記事:	https://qiita.com/deaikei/items/8615362d320c76e2ce0b
参考ソース:	https://gist.github.com/yasaichi/254a060eff56a3b3b858
-----------------------------------------------------------------------------"""



class Cluster:
    """
    1つのクラスタに関する情報を持ち、正規分布を仮定した時の尤度やBICの計算を行うクラス

    クラス変数 一覧
    numpy_matrix    __data          クラスタ内のデータ(行にサンプル，列に特徴変数)
    int             __size          クラスタ内のデータ数
    int             __ndim          クラスタ内のデータの次元数(特徴変数の数)
    numpy_matrix    __mu            クラスタ内のデータに関する平均ベクトル(サイズ:1×特徴変数の数)
    numpy_matrix    __sigma         クラスタ内のデータに関する分散共分散行列(サイズ:特徴変数の数×特徴変数の数)
    int             __num_of_param  ガウス分布モデルのパラメータ数
    """

    def __init__(self,X,covariance=False):
        """
        コンストラクタ
        
        Parameters
        ----------
        Cluster         self       : 本オブジェクト
        numpy_2d_array  X          : サンプル数*特徴ベクトル次元の配列, i.e. 行にサンプル，列に特徴変数
                                     shape=(n_samples, n_features)
        boolean         covariance : 分散共分散行列における共分散も計算するか否か
                                     default: False -> i.e. 分散共分散行列の対角成分以外を0にする
        """

        self.__data 	= np.matrix(X)								# クラスタ内のデータ
        self.__size 	= self.__data.shape[0]						# クラスタ内のデータ数
        self.__ndim 	= self.__data.shape[1]						# データの次元
        self.__mu 	    = np.matrix(np.average(self.__data,axis=0)) # 平均ベクトル

        # 分散共分散行列の計算
        if self.__size <= 1:
            # クラスタ内のデータ数が1つなら分散共分散行列の要素は全てゼロ
            self.__sigma = np.matrix(np.zeros((self.__ndim,self.__ndim)))
        else:
            # クラスタ内のデータ数が複数なら
            if covariance:
                # 共分散も計算する時
                self.__sigma 		= np.matrix(np.cov(self.__data.T))
            else:
                # 共分散は使用しなく、分散のみの情報を利用する場合は、分散共分散行列の対角成分のみを抽出する
                self.__sigma 		= np.matrix(np.cov(self.__data.T) * np.eye(self.__ndim))
        
        # ガウス分布モデルのパラメータ数
        if covariance:
            self.__num_of_param 	= self.__ndim * (self.__ndim + 3) / 2 # 平均ベクトル(__mu)の次元数と分散共分散行列(__sigma)の上三角成分の数の合計数(対称行列なので下三角成分の数は含めない)
        else:
            self.__num_of_param 	= self.__ndim *  2 # 平均ベクトル(__mu)の次元数と分散共分散行列(__sigma)の対角成分の数の合計
        
        # ここで __mu と __sigma はこのクラスタ内のデータにより、最尤推定によって取得されたパラメータのことを指す
	

    def _checkRank(self):
        """
        分散共分散行列のランク落ちのチェックメソッド

        Parameters
        ----------
        self : 本オブジェクト
        """
        print("データの次元:		 {} ".format(self.__ndim))
        print("分散共分散行列のランク: {}".format(np.linalg.matrix_rank(self.__sigma)))
        

    def _logLikelihood(self):
        """
        対数尤度を計算

        Parameters
        ----------
        self : 本オブジェクト

        Returns
        -------
        尤度
        """
        # クラスタ内のデータ数が1つの時は確率1を取ると仮定するので、対数尤度は0
        if self.__size <= 1:
            return 0

        # 以下の二つのどちらかで計算
        # 1つめは尤度(同時確率)を計算してから対数を取る つまり、掛け算した後に対数を取る
        # 2つめはそれぞれの確率を計算し、それぞれ対数を取っている　最後に対数をとったそれらの和を取る
        # これらの二つは指数法則対数法則より等価であるが、ほんのすこ〜〜〜〜〜〜しだけ誤差が発生する(1の方が精密な精度で計算する)

        # 1.
        # 自作のメソッドを用いる場合
        # ランク落ちとかでコードのチューニングを行う場合はこっち使う方が良い
        likelihood = 1
        for vector in self.__data:
            likelihood *= self._mnd(vector) 	# 自作のメソッドを使用しながら尤度(同時確率)を計算していく
        return np.log(likelihood) 	        # 最後に対数を取る

        # 2.
        # ライブラリのメソッドを使用
        # ランク落ちなどの場合、エラーが現れる場合があるので注意
        # likelihood = stats.multivariate_normal.logpdf(self.__data, np.array(self.__mu).reshape(-1), self.__sigma) # ライブラリを用いると各ベクトルに対する対数が取られた確率が要素となる配列が返ってくる
        # return np.sum(likelihood) # 最後に足す

    def _bic(self):
        """
        bicの計算

        Parameters
        ----------
        self : 本オブジェクト

        Returns
        -------
        bic
        """
        return -2 * self._logLikelihood() + np.log(self.__size) * self.__num_of_param


    def _mnd(self,x):
        """多次元正規分布による確率を計算

        Parameters
        ----------
                        self : 本オブジェクト
        numpy_1d_array  x    : 1つのデータ shape=(n_features)

        Returns
        -------
        多次元正規分布による確率
        """
        # 行列演算を容易にできるようにmatrix型へ変換
        x = np.matrix(x) 				# 対象ベクトル
        # print(self.__sigma)
        # self._checkRank()
        sigma_inv = self.__sigma.I				# 普通の逆行列
        # sigma_inv = np.linalg.pinv(self.__sigma) # 一般化逆行列(擬似逆行列) ランク落ちした時の逆行列を求めるときに有効 しかし以下の計算の時行列式で0になるので意味なし

        a = np.sqrt(np.linalg.det(self.__sigma)*(2*np.pi)**self.__ndim)
        b = np.linalg.det(-0.5*(x-self.__mu) * sigma_inv *(x-self.__mu).T)
        return np.exp(b)/a
    


    """
    Getter 
    """

    # クラスタ内のデータ
    def _data(self):
        return self.__data
    
    # クラスタ内のデータ数
    def _size(self):
        return self.__size
    
    # クラスタ内のデータにおける平均ベクトル
    def _mu(self):
        return self.__mu
    
    # クラスタ内のデータにおける分散共分散行列
    def _sigma(self):
        return self.__sigma

    # ガウス分布モデルのパラメータ数
    def _num_of_param(self):
        return self.__num_of_param



class ClusterOfParentChildren:
    """
    一つのクラスタを2分割する前とした後のそれぞれの情報を持つクラス
    ここでは，2分割する前をParentと呼び，2分割した後の2つのクラスタをChildrenと呼ぶ

    クラス変数
    boolean __covariance 分散共分散行列における共分散も計算するか否か
    Cluster __parent     ClusterクラスからParent用にインスタンス生成したもの
    Cluster __child_1    ClusterクラスからChildrenの片方用にインスタンス生成したもの
    Cluster __child_2    ClusterクラスからChildrenのもう片方用にインスタンス生成したもの
    """

    def __init__(self,X,covariance=False):
        """
        コンストラクタ

        Parameters
        ----------
                        self      : 本オブジェクト
        numpy_2d_array  X         : サンプル数*特徴ベクトル次元の配列, i.e. 行にサンプル，列に特徴変数
                                    shape=(n_samples, n_features)
        boolean         covariance: 分散共分散行列における共分散も計算するか否か
                                    default: False -> i.e. 分散共分散行列の対角成分以外を0にする
        """

        self.__covariance = covariance

        self.__parent = Cluster(X, self.__covariance)
        child_class_list = self._makeChildren() # 1つのクラスタを2つに分割
        self.__child_1 = child_class_list[0]
        self.__child_2 = child_class_list[1]

    def _makeChildren(self):
        """1つのクラスタを2つに分割

        Parameters
        ----------
        self :       本オブジェクト

        Returns
        -------
        child_list : 分割したそれぞれのクラスタに関するクラスを要素にもつリスト
        """
        k=2
        pred = KMeans(n_clusters=k).fit_predict(self.__parent._data())
        # print(pred) # クラスタ結果のラベルを出力(確認用)

        child_list = []
        # i を 0 ~ 1 に 変化(ここでiはk-meansで得られたクラスタラベルを示す)
        for i in range(k):
            child = self.__parent._data()[pred == i] # ラベルがiであるデータを取り出す -> child(子クラスタのひとつ)
            child_list.append(Cluster(child, self.__covariance))

        return child_list
	

    def _childrenBic(self):
        """2つに分割したクラスタに関するBICを計算する

        Parameters
        ----------
        self :       本オブジェクト

        Returns
        -------
        bic : 2つに分割したクラスタに関するBIC
        """
        # この辺は参考文献を頼りに！
        if np.linalg.det(self.__child_1._sigma()) == 0 and np.linalg.det(self.__child_2._sigma()) == 0:
            alpha = 0.5
        else:
            beta = np.linalg.norm(self.__child_1._mu() - self.__child_2._mu()) / np.sqrt(np.linalg.det(self.__child_1._sigma()) + np.linalg.det(self.__child_2._sigma()))
            alpha = 0.5 / stats.norm.cdf(beta) # stats.norm.cdf() -> 正規分布の累積分布関数を求めるやつだよ

        # 子クラスタに関するBICを返すよ
        return -2 * (self.__parent._size() * np.log(alpha) + self.__child_1._logLikelihood() + self.__child_2._logLikelihood()) + 2 * (self.__parent._num_of_param()) * np.log(self.__parent._size())
    

    """
    Getter 
    """

    # Parent
    def _parent(self):
        return self.__parent
    
    # Child_1
    def _child_1(self):
        return self.__child_1
    
    # Child_2
    def _child_2(self):
        return self.__child_2


"""-----------------------------------------------------------------------------"""


def xMeans(X, covariance=False):
    """
    x-meansを実行する関数
    幅優先探索のように実行していく

    Parameters
    ----------
    numpy_2d_array  X          : サンプル数*特徴ベクトル次元の配列, i.e. 行にサンプル，列に特徴変数
                                 shape=(n_samples, n_features)
    boolean         covariance : 分散共分散行列における共分散も計算するか否か
                                 default: False -> i.e. 分散共分散行列の対角成分以外を0にする

    Returns
    -------
    numpy_2d_array  cluster_centers : クラスタ数*特徴ベクトル次元の配列, i.e. 行に各クラスタ重心，列に特徴変数
                                      shape=(n_cluster, n_features)
                                      クラスタの重心を表す変数
    numpy_1d_array  labels          : サンプル数の要素を持つ配列, shape [n_samples,]
                                      Index of the cluster each sample belongs to.
    """

    q = [] 					# 幅優先を行うためのキュー
    layer_counter = 1 		# 階層カウンター
    layer_counter_q = [] 	# 階層カウンター保存用のキュー

    cluster_list = [] 		# 確定したクラスタを保存するリスト

    # 最初は1つのクラスタとしてキューに追加
    q.append(X)
    layer_counter_q.append(layer_counter)


    # キューの中がなくなるまでループ
    while True:
        
        if len(q) == 0:
            # キューの中がからならばクラスタリング終了
            break
        vectors = q.pop(0) # キューから対象クラスタを取り出す
        layer_counter = layer_counter_q.pop(0)

        if len(vectors) <= 1: # クラスタ内のデータが1つ以下ならそれは一つのクラスタとする
            cluster_list.append(vectors)
            continue

        instans_of_vectors = ClusterOfParentChildren(vectors, covariance)
        
        # 親と子のBICを比較し、親が大きければ子をキューに追加　親が小さければ親のクラスタを一つのクラスタとして確定させる
        # print(instans_of_vectors._parent()._bic(), instans_of_vectors._childrenBic()) # BIC確認用
        if instans_of_vectors._parent()._bic() > instans_of_vectors._childrenBic():
            q.append(instans_of_vectors._child_1()._data())
            q.append(instans_of_vectors._child_2()._data())
            layer_counter_q.append(layer_counter+1)
            layer_counter_q.append(layer_counter+1)
        else:
            cluster_list.append(vectors)

    # クラスタリング終了

    # クラスタの重心を計算
    clusters_centers = np.zeros((len(cluster_list),vectors.shape[1]))
    for index,cluster in enumerate(cluster_list):
        clusters_centers[index] = np.array(Cluster(cluster)._mu())

    # データの上から順にクラスタラベルを決定する
    label = np.zeros(len(X))
    for i, vector in enumerate(X):
        for cluster_label, cluster in enumerate(cluster_list):
            if vector in cluster:
                label[i] = cluster_label
                break

    return clusters_centers, label



"""-----------------------------------------------------------------------------"""


class XMeans:
    """
    x-meansに関するクラス

    クラス変数
    numpy_2d_array  __cluster_centers           クラスタ数*特徴ベクトル次元の配列, i.e. 行に各クラスタ重心，列に特徴変数
                                                shape=(n_cluster, n_features)
                                                クラスタの重心を表す変数
    Cluster         __cluster_centers_labels    ClusterクラスからParent用にインスタンス生成したもの
    numpy_1d_array  __labels                    サンプル数の要素を持つ配列, shape [n_samples,]
                                                Index of the cluster each sample belongs to.
    """
    

    def __init__(self):

        self.__cluster_centers = None        # クラスタ重心
        self.__cluster_centers_labels = None # クラスタ重心のラベル
        self.__labels = None                 # サンプルのラベル(サンプル順)

    
    def fit(self, X, covariance=False):
        """
        Compute x-means clustering.

        Parameters
        ----------
                        self       : 本オブジェクト
        numpy_2d_array  X          : サンプル数*特徴ベクトル次元の配列, i.e. 行にサンプル，列に特徴変数
                                     shape=(n_samples, n_features)
        boolean         covariance : 分散共分散行列における共分散も計算するか否か boolean型
                                     default: False -> i.e. 分散共分散行列の対角成分以外を0にする
        
        returns
        -------
                        self       : 本オブジェクト
        """

        self.__cluster_centers, self.__labels = xMeans(X, covariance)

        clf = KNeighborsClassifier(1).fit(X, self.__labels)
        self.__cluster_centers_labels = clf.predict(self.__cluster_centers)

        return self

    def fit_predict(self, X, covariance=False):
        """
        与えられたデータに基づいてx-meansにおけるクラスタリングを実行し，クラスタラベルを返す

        Parameters
        ----------
                        self       : 本オブジェクト
        numpy_2d_array  X          : サンプル数*特徴ベクトル次元の配列, i.e. 行にサンプル，列に特徴変数
                                     shape=(n_samples, n_features)
        boolean         covariance : 分散共分散行列における共分散も計算するか否か boolean型
                                     default: False -> i.e. 分散共分散行列の対角成分以外を0にする

        Returns
        -------
        numpy_1d_array  labels     : サンプル数の要素を持つ配列, shape [n_samples,]
                                     Index of the cluster each sample belongs to.
        """

        return self.fit(X, covariance).__labels
    

    def _check_is_fitted(self):
        """
        fitされているかのチェック

        Parameters
        ----------
        self :       本オブジェクト
        """

        if self.__cluster_centers is None:
            raise AttributeError("'XMeans' object has no attribute '__cluster_centers' yet.\n\tデータを'fit'させましょう")


    def _check_test_data(self, X):
        """
        テストデータの型(特徴ベクトルの次元数など)が不一致していないかのチェック

        Parameters
        ----------
                        self   : 本オブジェクト
        numpy_2d_array  X      : サンプル数*特徴ベクトル次元の配列, i.e. 行にサンプル，列に特徴変数
                                 shape=(n_samples, n_features)
        """
        X = np.array(X)
        n_features = X.shape[1]
        expected_n_features = self.__cluster_centers.shape[1]
        if not n_features == expected_n_features:
            raise ValueError("Incorrect number of features. "
                             "Got %d features, expected %d" % (
                                 n_features, expected_n_features))

        return X


    def predict(self, X):
        """
        すでにクラスタリングされているデータを元に，
        新規のデータがどのクラスタのラベルに属するかを推定する．
        推定方法は一番近いクラスタ重心を探索し，そのクラスタのラベルをつける

        Parameters
        ----------
                        self   : 本オブジェクト
        numpy_2d_array  X      : サンプル数*特徴ベクトル次元の配列, i.e. 行にサンプル，列に特徴変数
                                 shape=(n_samples, n_features)

        Returns
        -------
        numpy_1d_array  labels : サンプル数の要素を持つ配列, shape [n_samples,]
                                 Index of the cluster each sample belongs to.
        """

        # fitされているか確認
        self._check_is_fitted()

        # 入力されたデータの形式が正しいかチェック
        X = self._check_test_data(X)

        # 最も近い重心のラベルで予測分類
        clf = KNeighborsClassifier(1).fit(self.__cluster_centers, self.__cluster_centers_labels)
        return clf.predict(X)
    

    def score(self, X):
        """
        x-meansの精度の良さを算出する．
        まず、与えられたデータに対してどのクラスタに属するかを決定し，クラスタラベルを付与する．
        付与されたクラスラベルとそのラベルのクラスタ重心の距離を計算．
        同クラスタ内に関するその距離に基づくMSEを，そのクラスタにおけるスコアとする
        全てのクラスタにおけるスコアを和を返す

        Parameters
        ----------
                        self   : 本オブジェクト
        numpy_2d_array  X      : サンプル数*特徴ベクトル次元の配列, i.e. 行にサンプル，列に特徴変数
                                 shape=(n_samples, n_features)

        Returns
        -------
        float           score  : クラスタリングにおけるスコア
        """

        X_label = self.predict(X)
        score = 0
        for lbl in range(len(np.unique(X_label))):
            cluster_data = X[X_label == lbl]
            score += np.sqrt(np.sum((cluster_data - self.__cluster_centers[lbl])**2) / cluster_data.shape[0])
        
        return score
