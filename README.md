# sklearn-notes
These are my personal notes and some simple demo programs on using [sklearn](http://scikit-learn.org/stable/).

## Demo Programs

### [sklearn1.py](examples/sklearn1.py)

2-D [linear regression](http://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LinearRegression.html). 

### [sklearn2.py](examples/sklearn2.py)

[Principal component analysis](http://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html).

### [sklearn3.py](examples/sklearn3.py)

Training a [Support Vector Machine](http://scikit-learn.org/stable/modules/svm.html). 

### [sklearn4.py](examples/sklearn4.py)

Simple [KMeans](http://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html) clustering example.

### [sklearn5.py](examples/sklearn5.py)

Example of [Logistic Regression](http://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html) and [Polynomial Features](http://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.PolynomialFeatures.html)

### [sklearn6.py](examples/sklearn6.py)

Example of using [Lasso](http://scikit-learn.org/stable/modules/generated/sklearn.linear_model.Lasso.html) to find a sparse representation of data.

### [sklearn7.py](examples/sklearn7.py)

Example of using [Scale](http://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.scale.html) to normalize the data
allowing Logistic Regression to do a better job.

### [sklearn8.py](examples/sklearn8.py)

Example of using [Random Forest](http://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html). 

### [sklearn9.py](examples/sklearn9.py)

Example of using [GridSearchCV](http://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html). 
This is a revision of sklearn3.py which learns the best values for `gamma` and `C`.  

Note: The first parameter to GridSearchCV is the estimator.  It is important to include the `()` at the end. 
