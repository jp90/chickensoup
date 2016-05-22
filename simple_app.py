from spyre import server
import pandas as pd
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_moons, make_circles, make_classification
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.decomposition import PCA
from mpl_toolkits.mplot3d import Axes3D
from sklearn import manifold
from sklearn import svm, grid_search

names = ["Nearest Neighbors", "Linear/Poly SVM", "RBF SVM", "Decision Tree",
         "Random Forest", "AdaBoost", "Naive Bayes", "Linear Discriminant Analysis",
         "Quadratic Discriminant Analysis"]
classifiers = [
    KNeighborsClassifier(9,algorithm='auto',weights='distance'),
    SVC(kernel="linear", C=1),
    #SVC(kernel="poly",C=10,degree=3,coef0=1),
    SVC(kernel='rbf',gamma=1, C=2),
    DecisionTreeClassifier(max_depth=5),
    RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1),
    AdaBoostClassifier(),
    GaussianNB(),
    LinearDiscriminantAnalysis(),
    QuadraticDiscriminantAnalysis()]

# My data!!
proposals =pd.read_csv("/Users/jan-philippwolf/Documents/FOREX/JobMastersApplication/Horvath/Case/proposals.csv")
proposals.columns=['price', 'rep', 'creationdate', 'decisiondate', 'startdate','status','client']
psize = proposals['decisiondate'].size
# cleanup
proposals.loc[7372,('startdate')]='2004-12-31'
proposals.loc[20915,('startdate')]='2005-08-28'
proposals.loc[20845,('startdate')]='2005-08-28'
proposals.loc[35257,('startdate')]='2006-09-04'

def days_between(d1, d2):
    d1 = datetime.strptime(d1, "%Y-%m-%d")
    d2 = datetime.strptime(d2, "%Y-%m-%d")
    return (d2 - d1).days

t1=np.zeros(psize)
t2=np.zeros(psize)
t3=np.zeros(psize)
for i in range(0,psize):
    t1[i] = days_between(proposals['creationdate'].values[i],proposals['decisiondate'].values[i])
    t2[i] = days_between(proposals['creationdate'].values[i],proposals['startdate'].values[i])
    t3[i] = days_between(proposals['decisiondate'].values[i],proposals['startdate'].values[i])

# expand data set #proposals['t3'] = t3 # is redundant information
proposals['t1'] = t1
proposals['t2'] = t2
proposals = proposals.drop(proposals[proposals.t1<0].index)
proposals = proposals.drop(proposals[proposals.t2<0].index)
psize = proposals['decisiondate'].size
# which info do we consider?
info=['price','t1','t2','client','rep']
l=psize
myfeatures = proposals[info].values[0:l]
status = proposals['status'].values[0:l]

data = (myfeatures, status)

X, y = data
X = StandardScaler().fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1)
for name, clf in zip(names[2:3], classifiers[2:3]):
    print(clf)
    clf.fit(X_train, y_train)
    score = clf.score(X_test, y_test)
    print("Score using classifier",names[2:3],"is",score,".")

outcome=clf.predict(np.array([40,20,30,82,90]).reshape(1,-1))
print("Prediction is",outcome)

class SimpleSineApp(server.App):
    title = "Proposal Predictor"
    inputs = [{ "type":"slider",
                "key":"price",
                "label":"price",
                "value":40, "max":1000,
                "action_id":"sine_wave_plot"},
            {"type":"slider",
                "key":"t1",
                "label":"Delta creation to decision in days",
                "value":20, "max":365,
                "action_id":"sine_wave_plot"},
            {"type":"slider",
                "key":"t2",
                "label":"Delta creation to start in days",
                "value":20, "max":365,
                "action_id":"sine_wave_plot"},
            # {"type":'dropdown',
            #     "label": 'Client', 
            #     "options" : [ {"label": "Google", "value":"GOOG"},
            #                   {"label": "Yahoo", "value":"YHOO"},
            #                   {"label": "Apple", "value":"AAPL"}
            #                       ],
            #     "key": 'rep', 
            #     "action_id": "sine_wave_plot"},
            { "type":"text",
                "key":"client",
                "label":"Enter client id here",
                "value":"90", 
                "action_id":"sine_wave_plot"},
            { "type":"text",
                "key":"rep",
                "label":"Enter rep id here",
                "value":"128", 
                "action_id":"sine_wave_plot"}
            # {"type":'dropdown',
            #     "label": 'Representative', 
            #     "options" : [ {"label": "Max Mueller", "value":"mm"},
            #                   {"label": "Alexander Vogel", "value":"mddm"},
            #                   {"label": "Tim Tom", "value":"df"}
            #                       ],
            #     "key": 'rep', 
            #     "action_id": "sine_wave_plot"}                
            ]

    outputs = [{"type":"plot",
                "id":"sine_wave_plot"}]

    def getPlot(self, params):
        p = float(params['price'])
        t1 = float(params['t1'])
        t2 = float(params['t2'])
        rep=float(params['rep'])
        client=float(params['client'])
        outcome=clf.predict(np.array([p,t1,t2,rep,client]).reshape(1,-1))
        x = np.arange(0,2*np.pi,np.pi/150)
        y = np.sin(p*x)
        fig = plt.figure()
        splt1 = fig.add_subplot(1,1,1)
        if outcome==0:
           splt1.plot(x,y,'r')
        else:
           splt1.plot(x,y,'g')
        #splt1.add_patch(
        #patches.Rectangle(
        #(0.1, 0.1),   # (x,y)
        #0.5,          # width
        #0.5,          # height
        #), facecolor='red'
        #                )
        return fig

if __name__ == '__main__':
    app = SimpleSineApp()
    app.launch()