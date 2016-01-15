from sklearn.ensemble.partial_dependence import plot_partial_dependence
from sklearn.ensemble.partial_dependence import partial_dependence
import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt

selected_features = ['body_length','fb_published','gts','has_analytics','has_header', 'has_logo',
'name_length','num_order','num_payouts','org_facebook','org_twitter','sale_duration','sale_duration2',
'show_map','user_age','min_price','max_price', 'mean_price', 'total_revenue', 'total_tix_sold',
'total_tix_offered','A','C','E','G','M','N','U','num_caps_freq','email_.com','email_.gov',
'email_.org','email_.other','delivery_method_0.0','delivery_method_1.0','delivery_method_3.0','delivery_method_nan',
'state_GREATER_LONDON','state_FL','state_LONDON','state_GT_LON','state_DE','state_BIRMINGHAM','state_PA',
'state_NV','state_NH','state_GA','state_ENGLAND','country_US','country_IE','country_FR','country_CA',
'country_GB','country_AU','country_ES','country_NL','country_DE','country_VN','country_NZ','country_PK',
'country_MA','country_A1','country_other','previous_payout']

selected_features = np.array(selected_features)

features = [np.where(selected_features == 'total_tix_sold')[0][0], np.where(selected_features == 'mean_price')[0][0]]


fig, axs = plot_partial_dependence(gb4000_clf, X_train, features, feature_names = selected_features,
                                   n_jobs = -1, grid_resolution = 100)
fig.suptitle('Partial dependence of fraud detection features')
plt.subplots_adjust(top=0.9)  # tight_layout causes overlap with suptitle
#fig.tight_layout()

plt.show()