# XMeans

### 使用言語と必要なライブラリ
- Python3
    - numpy
    - scipy
    - scikit-learn


## 使用方法

```
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
```

共分散も計算するか否かも引数(covariance)で指定できる　指定しなければデフォルトでは計算しない  
参考論文では共分散を計算しなくとも同程度の結果になったらしい  
共分散も計算しちゃうとランク落ちしがちだからオススメはしない  
でもでも、データによっては共分散を計算しなくともランク落ちしちゃうから、その時はどんまい.ランク落ち時の対策法はネットにあるので調べましょう。  
(例えば似ているデータが複数ある時や次元が高すぎる時)


参考論文:	http://www.rd.dnc.ac.jp/~tunenori/doc/xmeans_euc.pdf  
参考記事:	https://qiita.com/deaikei/items/8615362d320c76e2ce0b  
参考ソース:	https://gist.github.com/yasaichi/254a060eff56a3b3b858  