import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score

def classification(X_train, y_train, X_test, y_test):
    n_exps = 10
    scores = np.zeros(n_exps)
    for i in range(n_exps):
        forest = RandomForestClassifier(n_estimators=30, max_depth=8)
        forest.fit(X_train, y_train)
        y_predict = forest.predict(X_test)
        score = f1_score(y_predict, y_test, average='macro')
        scores[i] = score
    return scores.mean(), scores.std()