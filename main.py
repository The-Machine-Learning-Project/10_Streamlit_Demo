#import libraries
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

from sklearn import datasets
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.decomposition import PCA

#header
st.title('Streamlit Example')

#subheader
st.write("""
# Explore different classifier
Which one is the best?
""")

#sidebar dataset selection
dataset_name = st.sidebar.selectbox("Select Dataset", ("Iris", "Breast Cancer", "Wine"))

#sidebar classifier selection
classifier_name = st.sidebar.selectbox("Select Classifier", ("KNN", "SVM", "Random Forest"))

#get_dataset function
def get_dataset(dataset_name):
    if dataset_name == "Iris":
        data = datasets.load_iris()
    elif dataset_name == "Breast Cancer":
        data = datasets.load_breast_cancer()
    else:
        data = datasets.load_wine()

    X = data.data
    y = data.target
    return X, y

#load dataset based on user selection
X,y = get_dataset(dataset_name)

#display dataset details
st.write("Shape of dataset:", X.shape)
st.write("Number of classes in dataset:", len(np.unique(y)))

#add_parameter_ui function
def add_parameter_ui(clf_name):
    params = dict()
    if clf_name == "KNN":
        K = st.sidebar.slider("K", 1,15)
        params["K"] = K
    
    elif clf_name == "SVM":
        C = st.sidebar.slider("C", .01, 10.0)
        params["C"] = C
    
    else:
        max_depth = st.sidebar.slider("max_depth", 2, 15)
        n_estimators = st.sidebar.slider("n_estimators", 2, 15)
        params["max_depth"] = max_depth
        params["n_estimators"] = n_estimators

    return params
    
#display the slider
params = add_parameter_ui(classifier_name)

#get_classifier_function
def get_classifier(clf_name, params):
    if clf_name == "KNN":
        clf = KNeighborsClassifier(n_neighbors=params["K"])

    elif clf_name == "SVM":
        clf = SVC(C=params["C"])
    else:
        clf = RandomForestClassifier(max_depth=params["max_depth"],
                                     n_estimators=params["n_estimators"], random_state = 1234
                                    )
    
    return clf

#retrieve selected classification model
clf = get_classifier(classifier_name, params)

#modelling
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = .2, random_state=1234)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
acc = accuracy_score(y_test, y_pred)

#display results
st.write(f"Classifier: {classifier_name}")
st.write(f"Accuracy: {acc}")

#plot results
pca = PCA(2)
X_projected = pca.fit_transform(X)

x1 = X_projected[:,0]
x2 = X_projected[:,1]

fig = plt.figure()
plt.scatter(x1, x2, c = y, alpha = .8, cmap = "viridis")
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.colorbar()

#plt.show()
st.set_option('deprecation.showPyplotGlobalUse', False)
st.pyplot()

# TODO
# - add more parameters
# - add other classifiers
#- improve UI

