from GB_model import run_gradient_boosted, run_gradient_boosted_evenFraud
from rf_model import run_rf_churn, run_rf_churn2
from clean_data import load_data, process_df
from get_features import get_features, train_test_equal_weight
import matplotlib.pyplot as plt
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
from sklearn.metrics import average_precision_score
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from sklearn import grid_search
from sklearn.cross_validation import train_test_split
import numpy as np 

def get_metrics(clf, X_test, y_test):
	if clf == gb4000_clf or clf == gb1000_clf:
		l = 'GB w/ {} Est'.format(clf.n_estimators)
	elif clf == gbEF_clf:
		l = 'GB w/ Even Dist'
	elif clf == rf1_clf:
		l = 'RF Model'
	else:
		l = 'RF with Even Dist'
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
	plt.plot(fpr, tpr, label = '{},AUC:{}'.format(l, AUC))
	return recall, AUC, precision, AUC2



def models_combined():
	df = load_data()
	df = process_df(df)
	X, y = get_features(df)

	print "running gradient boosted model with 4000 estimators..."
	gb4000_clf, gb4000_Xtest, gb4000_ytest = run_gradient_boosted(X, y, n_estimators = 4000)

	print "running gradient boosted model with 1000 estimators..."
	gb1000_clf, gb1000_Xtest, gb1000_ytest = run_gradient_boosted(X, y, n_estimators = 1000)
	
	print "running random forest model..."
	rf1_clf, rf1_Xtest, rf1_ytest = run_rf_churn(X,y)
	
	df = load_data()
	df = process_df(df)
	train_df, test_df = train_test_equal_weight(df)
	X_train, y_train = get_features(train_df)
	X_test, y_test = get_features(test_df)
	
	print "running gradient boosted model with even fraud and non-fraud..."
	gbEF_clf, gbEF_Xtest, gbEF_ytest = run_gradient_boosted_evenFraud(X_train, y_train, X_test, y_test)
	
	print "running random forest model with even fraud and non-fraud..."
	rf2_clf, rf2_Xtest, rf2_ytest = run_rf_churn2(X_train, y_train, X_test, y_test)

	return gb4000_clf, gb4000_Xtest, gb4000_ytest, gb1000_clf, gb1000_Xtest, gb1000_ytest, gbEF_clf, gbEF_Xtest, gbEF_ytest, rf1_clf, rf1_Xtest, rf1_ytest, rf2_clf, rf2_Xtest, rf2_ytest

if __name__ == '__main__':
	gb4000_clf, gb4000_Xtest, gb4000_ytest, gb1000_clf, gb1000_Xtest, gb1000_ytest, gbEF_clf, gbEF_Xtest, gbEF_ytest, rf1_clf, rf1_Xtest, rf1_ytest, rf2_clf, rf2_Xtest, rf2_ytest = models_combined()
	gb4000_recall, gb4000_AUC, gb4000_precision, gb4000_AUC2 = get_metrics(gb4000_clf, gb4000_Xtest, gb4000_ytest)
	gb1000_recall, gb1000_AUC, gb1000_precision, gb1000_AUC2 = get_metrics(gb1000_clf, gb1000_Xtest, gb1000_ytest)
	gbEF_recall, gbEF_AUC, gbEF_precision, gbEF_AUC2 = get_metrics(gbEF_clf, gbEF_Xtest, gbEF_ytest)
	rf1_recall, rf1_AUC, rf1_precision, rf1_AUC_precis_recall = get_metrics(rf1_clf, rf1_Xtest, rf1_ytest)
	rf2_recall, rf2_AUC, rf2_precision, rf2_AUC_precis_recall = get_metrics(rf2_clf, rf2_Xtest, rf2_ytest)
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
	plt.show()
