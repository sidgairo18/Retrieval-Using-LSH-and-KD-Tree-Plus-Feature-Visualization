# Locally Sensitive Hashing

The main idea behind Locally Sensitive Hashing is to create a hash such that we can find the nearest similar neighbours without having to compare every pair of data.

## Algorithm

The local sensitivity of a hash function is described with 4-tuple (d1, d2, p1, p2) where:

- d1 is the maximum distance between two similar samples.
- d2 is the minimum distance between two dissimilar samples (d2 = c * d1)
- p1 is the least probability of collision for similar samples
- p2 is the largest probability of collision for dissimilar samples

To define a hash function, we maximize p1 and minimize p2.

<br>
In practice, it is not possible to come up with a single hash function with acceptably high true-positive probability and acceptably low false-positive probability.

We use multiple hash functions in an AND-OR structure to amplify p1 and p2.

Let's say we have r hash functions.

### AND Relation

We say that two samples are similar only if the they are considered to be similar by all the r hash functions.

We note that p1 and p2 get amplified to p1^r and p2^r respectively.

                    x1     x2     x3     x4     x5
                     ----------------------------------
                    |
                h1  |  1      .      1      .      .        r=0
    hash fns    h2  |  0      .      0      .      .        r=1
                h3  |  1      .      1      .      .        r=2


Since all the functions consider x1 and x3 to be similar, then only they are considered to be similar.

However, assuming that p1 is quite high to begin with, say around 0.9, and p2 is quite low to begin with, say, around 0.1. The p1^r value will be impacted much less that the p2^r value.

### OR Relation

We say that two samples are similar only if the they are considered to be similar by any of the r hash functions.

We note that p1 and p2 get amplified to [1 - (1 - p1)^b] and [1 - (1 - p2)^b] respectively.

                    x1     x2     x3     x4     x5
                     ----------------------------------
                    |
                h1  |  0      .      1      .      .        r=0
    hash fns    h2  |  0      .      0      .      .        r=1
                h3  |  0      .      1      .      .        r=2


Since h2 considers x1 and x3 to be similar, they are considered to be similar.

### AND-OR Relation

If we use both the relations, our amplified probabilities get updated to [1 - (1 - p1^r)^b] and [1 - (1 - p2^r)^b]

                                data samples
                       x1     x2     x3     x4     x5
                     ----------------------------------
                    |
                h1  |  1      .      1      .      .        b=0  r=0
                h2  |  0      .      0      .      .        b=0  r=1
    hash        h3  |  1      .      1      .      .        b=0  r=2
    functions       |
                h4  |  .      1      .      1      .        b=1  r=0
                h5  |  .      1      .      1      .        b=1  r=1
                h6  |  .      0      .      0      .        b=1  r=2
                    |
                h7  |  1      .      .      .      1        b=2  r=0
                h8  |  1      .      .      .      1        b=2  r=1
                h9  |  1      .      .      .      1        b=2  r=2
                    |

The given set of hash functions consider x1 and x3, x2 and x4, and x1 and x5 to be similar.