from ggplot import *
import numpy as np
from sklearn import datasets
import pandas as pd
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from bokeh.plotting import figure, show

# Involves visualizing MNIST dataset by applying PCA and TSNE.

def load_data():
    digits = datasets.load_digits()
    X_test = np.zeros(shape=(digits.images.shape[0], digits.images.shape[1]*digits.images.shape[2]))
    for i in range(digits.images.shape[0]):
        X_test[i, :] = digits.images[i, :].flatten()
    Y_test = digits.target
    print(X_test.shape, Y_test.shape)
    return X_test, Y_test

def plot(X_test, rndperm, df):
    # chart = ggplot(df, aes(x='c1', y='c2', color='label') ) \
    #         + geom_point(size=75,alpha=0.8) \
    #         + ggtitle("Result on applying pca first and tsne later")
    # print(chart)
    images = []
    for i in range(rndperm.shape[0]):
        images.append(X_test[rndperm[i], :].reshape(8, 8))
    p = figure(x_range=(np.min(df['c1'])-30, np.max(df['c1'])+30), y_range=(np.min(df['c2'])-30,
                        np.max(df['c2'])+30), plot_width=950, plot_height=950)
    p.image(image=images, x=df['c1'], y=df['c2'], dw=1, dh=1)
    show(p)

def apply_pca(data):
    pca = PCA(n_components=50)
    pca_result = pca.fit_transform(data)
    return pca_result

def apply_tsne(data):
    tsne = TSNE(n_components=2, verbose=1, perplexity=40, n_iter=500)
    tsne_result = tsne.fit_transform(data)
    return tsne_result

if __name__== '__main__':

    # convert matrix to PandasDataFrame
    X_test, Y_test = load_data()
    feat_cols = [ 'pixel'+str(i) for i in range(X_test.shape[1]) ]
    df = pd.DataFrame(X_test, columns=feat_cols)
    df['label'] = Y_test
    df['label'] = df['label'].apply(lambda i: str(i))

    rndperm = np.random.permutation(df.shape[0])
    pca_result = apply_pca(df[feat_cols].values)
    tsne_result = apply_tsne(pca_result[rndperm])
    df_tsne = df.loc[rndperm,:].copy()
    df_tsne['c1'] = tsne_result[:, 0]
    df_tsne['c2'] = tsne_result[:, 1]
    plot(X_test, rndperm, df_tsne)