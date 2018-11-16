from ggplot import *
import numpy as np
import random
from sklearn import datasets
from PIL import Image
import pandas as pd
import pickle
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from bokeh.plotting import figure, show

def load_data():
    label_file = open('bam_2_0_image_style_labels.pkl','r')
    label_data = pickle.load(label_file)
    label_keys = label_data.keys()
    random.shuffle(label_keys)
    label_data = [(key, label_data[key]) for key in label_keys]
    Y_test = np.zeros((5000,1))
    X_test = np.zeros((5000,1000))
    I_test = []
    row = 0
    for image_name, label in label_data:
        print(row)
        Y_test[row,:] = label
        X_test[row,:] = np.load('../../final_features2_pca/' + image_name + '.npy')
        #print('/scratch/bam_subset_2_0/' + image_name)
        I_test.append(np.rot90(np.array(Image.open('/scratch/bam_subset_2_0/' + image_name).resize((100,100), Image.BICUBIC).convert('RGBA')), 2))
        row = row + 1
        if row == 5000:
            break
    return X_test, Y_test, I_test

def plot(X_test, I_test, df):
    chart = ggplot(df, aes(x='c1', y='c2', color='label') ) \
            + geom_point(size=75,alpha=0.8) \
            + ggtitle("Result on applying pca first and tsne later")
    # print(chart)
    p = figure(x_range=(np.min(df['c1']), np.max(df['c1'])),
               y_range=(np.min(df['c2']), np.max(df['c2'])),
               plot_width=950, plot_height=950)
    p.image_rgba(image=I_test, x=df['c1'], y=df['c2'], dw=1, dh=1)
    show(p)

def apply_pca(data, pca_components):
    pca = PCA(n_components=pca_components)
    pca_result = pca.fit_transform(data)
    return pca_result

def apply_tsne(data):
    tsne = TSNE(n_components=2, verbose=1, perplexity=40, n_iter=500)
    tsne_result = tsne.fit_transform(data)
    return tsne_result

def visualize_features(X_test, Y_test, I_test, pca_components):
    # convert matrix to PandasDataFrame
    feat_cols = [ 'pixel'+str(i) for i in range(X_test.shape[1]) ]
    df = pd.DataFrame(X_test, columns=feat_cols)
    df['label'] = Y_test
    df['label'] = df['label'].apply(lambda i: str(i))

    rndperm = np.random.permutation(df.shape[0])
    pca_result = apply_pca(df[feat_cols].values, pca_components)
    tsne_result = apply_tsne(pca_result[rndperm])
    df_tsne = df.loc[rndperm,:].copy()
    I_copy = []
    for i in range(rndperm.shape[0]):
        I_copy.append(I_test[rndperm[i]])

    df_tsne['c1'] = tsne_result[:, 0]
    df_tsne['c2'] = tsne_result[:, 1]
    plot(X_test, I_copy, df_tsne)

if __name__== '__main__':

    X_test, Y_test, I_test = load_data()
    visualize_features(X_test, Y_test, I_test, 50)