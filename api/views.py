
from api import app
import os
import pandas as pd
import json
from flask import jsonify
from flask import render_template
from sklearn import preprocessing
from sklearn import decomposition
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from flask import url_for
import sklearn.cross_validation as cv
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
import sklearn.metrics as metrics


def get_abs_path():
    return os.path.abspath(os.path.dirname(__file__))


def get_data():
    f_name = os.path.join(get_abs_path(), 'data', 'breast-cancer-wisconsin.csv')
    columns = ['code', 'clump_thickness', 'size_uniformity', 'shape_uniformity',
               'adhesion', 'cell_size', 'bare_nuclei', 'bland_chromatin',
               'normal_nuclei', 'mitosis', 'class']
    df = pd.read_csv(f_name, sep=',', header=None, names=columns, na_values='?')
    return df.dropna()


@app.route('/')
def index():
    df = get_data()
    X = df.ix[:,(df.columns != 'class') & (df.columns != 'code')].as_matrix()
    y = df.ix[:, df.columns == 'class'].as_matrix()
    # Scale
    scaler = preprocessing.StandardScaler().fit(X)
    scaled = scaler.transform(X)
    # PCA
    pcomp = decomposition.PCA(n_components=2)
    pcomp.fit(scaled)
    components = pcomp.transform(scaled)
    var = pcomp.explained_variance_ratio_.sum() # View w/ Debug
    # KMeans
    model = KMeans(init='k-means++', n_clusters=2)
    model.fit(components)
    # Plot
    fig = plt.figure()
    plt.scatter(components[:,0], components[:,1], c=model.labels_)
    centers = plt.plot(
        [model.cluster_centers_[0,0], model.cluster_centers_[1,0]],
        [model.cluster_centers_[1,0], model.cluster_centers_[1,1]],
        'kx', c='Green'
    )
    # Increase size of center points
    plt.setp(centers, ms=11.0)
    plt.setp(centers, mew=1.8)
    # Plot axes adjustments
    axes = plt.gca()
    axes.set_xlim([-7.5, 3])
    axes.set_ylim([-2, 5])
    plt.xlabel('PC1')
    plt.ylabel('PC2')
    plt.title('Clustering of PCs ({:.2f} % var. Explained'.format(
        var * 100
    ))
    # Save fig
    fig_path = os.path.join(get_abs_path(),'static', 'tmp', 'cluster.png')
    fig.savefig((fig_path))
    return render_template('index.html',
                           fig=url_for('static',
                                       filename='tmp/cluster.png'))


@app.route('/d3')
def d3():
    df = get_data()
    X = df.ix[:, (df.columns != 'class') & (df.columns != 'code')].as_matrix()
    y = df.ix[:, df.columns == 'class'].as_matrix()
    # Scale
    scaler = preprocessing.StandardScaler().fit(X)
    scaled = scaler.transform(X)
    # PCA
    pcomp = decomposition.PCA(n_components=2)
    pcomp.fit(scaled)
    components = pcomp.transform(scaled)
    var = pcomp.explained_variance_ratio_.sum() # View w/ Debug
    # KMeans
    model = KMeans(init='k-means++', n_clusters=2)
    model.fit(components)
    # Generate CSV
    cluster_data = pd.DataFrame({'pc1': components[:, 0],
                                 'pc2': components[:, 1],
                                 'labels': model.labels_})
    csv_path = os.path.join(get_abs_path(), 'static', 'tmp', 'kmeans.csv')
    cluster_data.to_csv(csv_path)
    return render_template('d3.html',
                           data_file=url_for('static',
                                             filename='tmp/kmeans.csv'))


@app.route('/prediction')
def prediction():
    df = get_data()
    features = df.ix[:, (df.columns != 'class') & (df.columns != 'code')].as_matrix()
    outcome = df.ix[:, df.columns == 'class'].as_matrix()
    x_train, x_test, y_train, y_test = cv.train_test_split(features, outcome, test_size=0.4, random_state=0)

    clf = RandomForestClassifier(n_estimators=1000, n_jobs=-1, bootstrap=True, warm_start=True, random_state=0)
    clf = clf.fit(x_train, y_train)

    y_true = y_test
    y_score = clf.predict_proba(x_test)[:, 1]
    fpr, tpr, thresholds = metrics.roc_curve(y_true, y_score, pos_label=4)
    roc_auc = metrics.auc(fpr, tpr)

    fig = plt.figure()
    plt.plot(fpr, tpr, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 0], [1, 1], 'k--')
    plt.xlim([-0.05, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Random Forest')
    plt.legend(loc="lower right")

    # Save fig
    fig_path = os.path.join(get_abs_path(),'static', 'tmp', 'roc_curve.png')
    fig.savefig((fig_path))
    return render_template('prediction.html',
                           fig=url_for('static',
                                       filename='tmp/roc_curve.png'))


@app.route('/api/v1/prediction_confusion_matrix')
def prediction_confusion_matrix():
    df = get_data()
    features = df.ix[:, (df.columns != 'class') & (df.columns != 'code')].as_matrix()
    outcome = df.ix[:, df.columns == 'class'].as_matrix()
    x_train, x_test, y_train, y_test = cv.train_test_split(features, outcome, test_size=0.4, random_state=0)

    clf = RandomForestClassifier(n_estimators=1000, n_jobs=-1, bootstrap=True, warm_start=True, random_state=0)
    clf = clf.fit(x_train, y_train)

    expected = y_test
    predicted = clf.predict(x_test)

    confusion_matrx = confusion_matrix(expected, predicted)
    TP = confusion_matrx[0, 0]
    TN = confusion_matrx[1, 1]
    FP = confusion_matrx[1, 0]
    FN = confusion_matrx[0, 1]

    matrix_df = pd.DataFrame([FP, TP, FN, TN], index=['fp', 'tp', 'fn', 'tn'], columns=['Random Forest'])
    data = json.loads(matrix_df.to_json())
    return jsonify(data)




@app.route('/head')
def head():
    df = get_data().head()
    data = json.loads(df.to_json())
    return jsonify(data)


