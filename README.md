# Wine_Classifier
To classify a set of samples into a certain
number of classes. You will first extract discriminative features from the data
samples, then build a classifier that uses such features to recognise what class
a given sample belongs to.
The Wine Dataset consists of 178
samples, each corresponding to a wine. Each wine is produced in the same
region in Italy, and each wine derives from one of three different cultivars (or
type of plant). We will treat the three different cultivars as classes.
Each sample contains 13 features/dimensions, corresponding to a chemical
constituent of the wine: 1) Alcohol, 2) Malic acid, 3) Ash, 4) Alkalinity of
ash, 5) Magnesium, 6) Total phenols, 7) Flavanoidsm, 8) Non
avanoid phenols,
9) Proanthocyanins, 10) Color intensity, 11) Hue, 12) OD280/OD315
of diluted wines, 13) Proline.
Now, we are neither wine experts nor do we know much (if anything)
about chemistry, but still we want to tell what cultivar each wine derives
from. The beauty of machine learning is that we can do that (to some
extent) without being an expert!
Training and testing We split the dataset in a 70/30 proportion to create
the training and testing sets. These sets are labelled, i.e. the class each
sample belongs to is known. You will need to use the training set to train
the classifier, and use the test set to check your code is working correctly,
as well as to report results. We also generated a private test set, which we
will use to auto-mark your code. The private test set will contain the same
features as the training and public test sets. We will not release the private
test set.
