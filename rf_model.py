from sklearn.ensemble import RandomForestClassifier
from sklearn.cross_validation import cross_val_score
from sklearn.grid_search import GridSearchCV
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
from sklearn.metrics import average_precision_score
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from sklearn.cross_validation import train_test_split
import matplotlib.pyplot as plt
import numpy as np


def run_rf_churn(X, y):
    #split data to 20% for testing
    X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.2)
    clf = RandomForestClassifier(random_state=2, n_estimators = 10000)
    clf.fit(X_train, y_train)
    # pred = clf.predict_proba(X_test)
    # pred2 = clf.predict(X_test)
    # fpr, tpr, thresholds = roc_curve(y_test, pred[:,1])
    # AUC = roc_auc_score(y_test, pred[:,1])
    # AUC2 = average_precision_score(y_test, pred[:,1])
    # recall = recall_score(y_test, pred2)
    # precision = precision_score(y_test, pred2)
    # plt.plot(fpr, tpr, label = 'Random Forest1 AUC = {}'.format(AUC))
    # print recall, AUC, precision, AUC2 
    return clf, X_test, y_test

def run_rf_churn2(X_train, y_train, X_test, y_test):
    clf = RandomForestClassifier(random_state=2, n_estimators = 10000)
    clf.fit(X_train, y_train)
    # pred = clf.predict_proba(X_test)
    # pred2 = clf.predict(X_test)
    # fpr, tpr, thresholds = roc_curve(y_test, pred[:,1])
    # AUC = roc_auc_score(y_test, pred[:,1])
    # AUC2 = average_precision_score(y_test, pred[:,1])
    # recall = recall_score(y_test, pred2)
    # precision = precision_score(y_test, pred2)
    # plt.plot(fpr, tpr)
    # v = np.linspace(0,1)
    # plt.plot(fpr, tpr, label = 'Random Forest2 AUC = {}'.format(AUC))
    # print recall, AUC, precision, AUC2 
    return clf, X_test, y_test

selected_features = np.array(['body_length', 'fb_published', 'gts', 'has_analytics', 'has_header', 'has_logo', 'name_length', 'num_order', 'num_payouts', 'org_facebook', 'org_twitter', 'sale_duration', 'show_map', 'user_age', 'min_price', 'max_price', 'mean_price', 'total_revenue', 'total_tix_sold', 'total_tix_offered', 'A', 'C', 'E', 'G', 'M', 'N', 'U', 'num_caps_freq', 'email_.com', 'email_.gov', 'email_.org', 'email_.other', 'delivery_method_0.0', 'delivery_method_1.0', 'delivery_method_3.0', 'delivery_method_nan', 'state_GREATER_LONDON', 'state_FL', 'state_LONDON', 'state_GT_LON', 'state_DE', 'state_BIRMINGHAM', 'state_PA', 'state_NV', 'state_NH', 'state_GA', 'state_ENGLAND', 'country_US', 'country_IE', 'country_FR', 'country_CA', 'country_GB', 'country_AU', 'country_ES', 'country_NL', 'country_DE', 'country_VN', 'country_NZ', 'country_PK', 'country_MA', 'country_A1', 'country_other', 'previous_payout'])
 
def plot_feature_importance(model):
    importances = model.feature_importances_
    # std = np.std([tree.feature_importances_ for tree in model.estimators_], axis=0)
    indices = np.argsort(importances)[::-1]
    #importances[indices]
    plt.figure()
    plt.title("Feature importances")
    plt.bar(range(len(indices)), importances[indices], color="b")#, yerr=std[indices], align="center")
    plt.xticks(range(len(indices)), selected_features[indices], rotation = 25)
    plt.tight_layout()
    plt.subplots_adjust(left=0.1, right=.9, top=0.9, bottom=0.15)
    plt.xlim([-0.1, 9.9])
    # plt.xlim([9.9, 19.9])
    plt.show()

if __name__ == '__main__':
	recall, AUC, precision, AUC_precis_recall = run_rf_churn(X,y)
	plt.show()