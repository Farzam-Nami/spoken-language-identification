# Methodology

## Audio corpus and preprocessing

The task is four-class spoken-language identification using approximately one-minute audiobook excerpts in German, Italian, Korean and Spanish. Each input folder is subdivided into Male and Female recordings; these labels describe the corpus and are excluded from the model inputs. The manifest records paths, file names, file sizes and MP3 durations. Mutagen checks metadata before full decoding. The recorded corpus has 720 clips, all with readable MP3 metadata. The original duration distribution has median 60.687 s and a maximum of 1534.24 s. The inclusive 50–70 s rule retains 655 clips.

Librosa loads mono waveforms at 16 kHz, trims leading and trailing low-energy regions at `top_db=25`, and normalizes each nonzero waveform by its maximum absolute amplitude:

$$y'_t = y_t / \max_j |y_j|.$$

SoundFile writes 16-bit PCM WAV files. These steps reduce variability in channel, sample rate, leading/trailing silence and peak amplitude, but are not explicit noise removal. Clips are not padded or truncated to a uniform duration, and no augmentation is implemented. Existing processed files are reused. The saved manifests indicate no preprocessing or feature extraction failures among the retained clips.

## Acoustic representation

The feature configuration uses 13 MFCCs, 2048 FFT samples and 512-sample hops at 16 kHz. Each frame-level feature is pooled over time by its mean and population standard deviation. For a feature sequence $z_t$ with $T$ frames:

$$\bar z = \frac{1}{T}\sum_t z_t,\qquad s_z = \sqrt{\frac{1}{T}\sum_t(z_t-\bar z)^2}.$$

| Feature group | Frame-level descriptors | Pooled dimensions |
|---|---|---:|
| Time domain | Zero-crossing rate, RMS | 4 |
| Spectral | Centroid, bandwidth, 85% rolloff, flatness, seven contrast bands | 22 |
| Cepstral | 13 MFCCs, 13 delta and 13 delta-delta coefficients | 78 |
| Chroma | 12 pitch-class descriptors | 24 |
| Total | 64 descriptors | 128 |

RMS summarizes local signal magnitude and zero-crossing rate captures sign changes. Centroid, bandwidth and rolloff describe spectral distribution; flatness and contrast capture spectral shape. MFCCs summarize the mel-scaled spectral envelope; deltas describe local dynamics. Chroma is a complementary pitch-class representation. Temporal pooling permits fixed-length inputs but sacrifices sequence-level order. The extracted table contains 655 rows, 128 numeric features and three metadata columns.

## Supervised models

A custom stratified split uses NumPy's random generator with seed 42 and selects the ceiling of 20% of each class for testing: 523 training and 132 test clips. Standardization uses the training mean and population standard deviation (with a small numerical stabilizer). SVD of the standardized training matrix yields the first six principal axes, which are reused for both partitions. Labels, gender and path columns are excluded from the model input.

### K-nearest neighbors

For a test point $x$, compute squared Euclidean distances to all training points. Predict by majority vote of the seven closest points. This local rule can represent nonlinear class boundaries without fitting a parametric decision function.

### Linear support vector machine (one-vs-rest)

Each language is treated as a binary class against the remaining languages. The weight vector is updated by stochastic gradient descent using hinge loss $\max(0,1-y(w^Tx+b))$ and L2 weight regularization. Prediction uses the largest class score. The actual run uses 25 epochs, learning rate 0.002, regularization 0.01, seed 42.

### Multi-layer perceptron

The MLP uses six input features, 16 hidden ReLU units and four softmax outputs. It is trained by explicit mini-batch forward and backward passes with softmax cross-entropy and L2 weight regularization. Actual configuration: 40 epochs, batch size 64, learning rate 0.01, regularization 0.02, seed 42. The recorded training accuracy reaches 0.9790.

## Unsupervised models

Clustering uses all 128 descriptors standardized over the complete corpus, not the six-dimensional supervised representation. Labels are not used for assigning clusters.

- **K-Means:** initialize from randomly selected data points (seed 42); alternate nearest-centroid assignment and centroid means, with empty clusters retaining their old centroids. Stop on `allclose` or 100 iterations. Compute within-cluster sum of squares for k=1 to 9 and choose k=7 visually.
- **Hierarchical:** initialize singleton clusters and merge the pair with minimum mean squared pairwise distance until four remain. This is agglomerative average linkage with squared distances. It is computationally expensive because pairs are searched explicitly.
- **Visualization:** project centered standardized features onto the leading two SVD components for visualization only. No t-SNE is executed.

## Evaluation definitions

For supervised models, the confusion matrix has true classes as rows and predicted classes as columns. Accuracy is the fraction correctly classified. Per-class precision, recall and F1 use the usual TP/FP/FN formulas, with undefined ratios set to zero. Macro averages weight each class equally; weighted averages weight by test support.

For clustering, purity is

$$purity = \frac{1}{N}\sum_k \max_c |C_k\cap L_c|.$$

The notebook's custom silhouette-like measure uses squared Euclidean distances, includes the point itself in the within-cluster average, and assigns zero contribution to singleton clusters. It therefore differs from the standard Euclidean silhouette (which excludes self-distance). The metric is retained exactly as implemented. Purity can favor larger k; comparing k=7 and k=4 is exploratory rather than a matched-k test.
