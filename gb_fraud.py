
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.cross_validation import cross_val_score
from sklearn.grid_search import GridSearchCV
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
from sklearn.metrics import average_precision_score
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from sklearn import grid_search
from sklearn.cross_validation import train_test_split

def run_gradient_boosted_gsearch(X,y):
	#split data to 20% for testing
	X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.2)
	#setup paramgrid for grid search
	param_grid = [{'learning_rate': [.08, .05, .1], 'n_estimators': [1000,4000,6000]}]
	GB = GradientBoostingClassifier()
	clf = GridSearchCV(GB, param_grid, verbose = 2, cv =10, n_jobs = -1) #10 k-folds
	clf.fit(X_train, y_train)
	pred = clf.predict_proba(X_test) #get back probabilities
	pred2 = clf.predict(X_test) #get back predictions
	fpr, tpr, thresholds = roc_curve(y_test, pred[:,1])
	#get the AUC
	AUC = roc_auc_score(y_test, pred[:,1])
	#get the AUC for precision and recall curve
	AUC2 = average_precision_score(y_test, pred[:,1])
	recall = recall_score(y_test, pred2)
	precision = precision_score(y_test, pred2)
	#plot AUC
	plt.plot(fpr, tpr, label = 'Random Forest AUC = {}'.format(AUC))
	v = np.linspace(0,1)
	plt.plot(v,v, linestyle = '--', color = 'b')
	plt.xlabel("False Postive Rate")
	plt.ylabel("True Postive Rate")
	plt.title('ROC Curve')
	plt.xlim(-0.2,1)
	plt.ylim(0,1.2)
	plt.axhline(1, color = 'k', linestyle = '--')
	plt.axvline(0, color = 'k', linestyle = '--')
	plt.legend()
	print recall, AUC, precision, AUC2 
	return clf, recall, AUC, precision, AUC2

def run_gradient_boosted(X, y, n_estimators = 5000):
	#split data to 20% for testing
	X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.2)
	clf = GradientBoostingClassifier(learning_rate = 0.05, n_estimators = n_estimators)
	clf.fit(X_train, y_train)
	# pred = clf.predict_proba(X_test) #get back probabilities
	# pred2 = clf.predict(X_test) #get back predictions
	# fpr, tpr, thresholds = roc_curve(y_test, pred[:,1])
	# #get the AUC
	# AUC = roc_auc_score(y_test, pred[:,1])
	# #get the AUC for precision and recall curve
	# AUC2 = average_precision_score(y_test, pred[:,1])
	# recall = recall_score(y_test, pred2)
	# precision = precision_score(y_test, pred2)
	# #plot AUC
	# plt.plot(fpr, tpr, label = 'Gradient Boosted {} Estimators AUC = {}'.format(n_estimators, AUC))
	# print recall, AUC, precision, AUC2 
	return clf, X_test, y_test

def run_gradient_boosted_evenFraud(X_train, y_train, X_test, y_test):
	clf = GradientBoostingClassifier(learning_rate = 0.05, n_estimators = 4000)
	clf.fit(X_train, y_train)
	# pred = clf.predict_proba(X_test) #get back probabilities
	# pred2 = clf.predict(X_test) #get back predictions
	# fpr, tpr, thresholds = roc_curve(y_test, pred[:,1])
	# #get the AUC
	# AUC = roc_auc_score(y_test, pred[:,1])
	# #get the AUC for precision and recall curve
	# AUC2 = average_precision_score(y_test, pred[:,1])
	# recall = recall_score(y_test, pred2)
	# precision = precision_score(y_test, pred2)
	# #plot AUC
	# plt.plot(fpr, tpr, label = 'Gradient Boosted (Even Fraud) AUC = {}'.format(AUC))
	# print recall, AUC, precision, AUC2 
	return clf, X_test, y_test

if __name__ == '__main__':
	clf, recall, AUC, precision, AUC2 = run_gradient_boosted_gsearch(X,y)
	best_params = clf.best_params_