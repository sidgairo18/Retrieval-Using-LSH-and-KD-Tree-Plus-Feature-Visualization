# t-distributed stochastic neighbor embedding

It is a nonlinear dimensionality reduction technique well-suited for embedding high-dimensional data for visualization in low
dimensional space of two- or three-dimensional point in such a way that similar objects are modeled by nearby points and
dissimilar objects are modeled by distant points with high probability.

## Algorithm
The t-SNE algorithm mainly comprises of two stages:
- First it constructs a probability distribution over pairs of high-dimensional objects in such a way that similar objects have a high
probability of being picked, whilst dissimilar points have an extremely small probability of being picked.
- Second, t-SNE defines a similar probability distribution over the points in the low-dimensional map, and it minimizes the [Kullback–Leibler
divergence](https://en.wikipedia.org/wiki/Kullback%E2%80%93Leibler_divergence) between the two distributions with respect to the locations 
of the points in the map.

### Stage-1 
- Given a set of N dimensional points x1,x2...xN t-SNE first computes the probability p<sub>ij</sub> that are proportional to the similarity of objects as follows:

  ![alt text](https://wikimedia.org/api/rest_v1/media/math/render/svg/2cc3ef3b4d237787cd82e5ef638d96d642a1e43d)
- Next p<sub>ij</sub> is calculated using the below formula:

  ![alt text](https://wikimedia.org/api/rest_v1/media/math/render/svg/a53cc5533bb4b3b8f18231c58df4e4215546a0fc)

### Stage-2
- In the second step t-SNE aims to learn a lower dimensional map y1,y2,y3...yN where dimension of yi < dimension of xi that reflects similarities in p<sub>ij</sub> as well as possible.
- To this end, it measure similarities q<sub>ij</sub> between two points in the map yi and yj using a very similar approach. q<sub>ij</sub> is defined as follows:

  ![alt text](https://wikimedia.org/api/rest_v1/media/math/render/svg/35a406737842de91ec9edc30efa559d52e6f52a3)

- The locations of the points yi in the map are detemined by minimizing [ Kullback–Leibler divergence](https://en.wikipedia.org/wiki/Kullback%E2%80%93Leibler_divergence) of the second distribution from the first one as follows:

    ![alt text](https://wikimedia.org/api/rest_v1/media/math/render/svg/cae779cfc3a41b382e68850f0381b6a6b7fdede7)
- The minimizing is performed using gradient descent. The result of this optimization is a map that reflects the similarities between the high-dimensional inputs well.

## Conclusions
- While t-SNE plots often seem to display clusters, the visual clusters can be influenced strongly by the chosen parameterization and therefore a good understanding of the parameters for t-SNE is necessary.
- t-SNE has been used for visualization in a wide range of applications, including computer security research, music analysis, cancer research, bioinformatics and biomedical signal processing.

## Existing Implementations
-  sklearn library in python contains this implementation: [sklearn.manifold.TSNE](http://scikit-learn.org/stable/modules/generated/sklearn.manifold.TSNE.html)