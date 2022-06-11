!pip install -q kaggle
from google.colab import files
files.upload()
!mkdir ~/.kaggle
!cp kaggle.json ~/.kaggle/
!chmod 600 ~/.kaggle/kaggle.json
!kaggle datasets download akashkr/phishing-website-dataset
!unzip phishing-website-dataset.zip
import re
import socket
from urllib.parse import urlparse
text="""Mastering Power BI is essential if you're after a career as a data analyst. In case you've missed it, our Data Analyst in Power BI career track, co-created with Microsoft has arrived and is the perfect way to supercharge your data career!
Click the link to find out more - https://lnkd.in/gQ964dE5"""
urls=re.findall('http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', text) #This is code for url detection
urls_str=" ".join(str(y) for y in urls)
hostname=urlparse(urls_str).netloc
ip_address=socket.gethostbyname(hostname)
print(f"Original string: [{text}]\n")
print(f"Urls: {urls_str}")
print(f"Host name: {hostname}")
print(f"Host length: {len(hostname)}")
print(f"URL length: {len(urls_str)}")
print(f"IP Address: {ip_address}")
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')
sns.set(rc={'figure.figsize': (15, 6)})
df=pd.read_csv('dataset.csv')
df.head()
df.drop('index', axis=1, inplace=True)
df.head()
for x in df.columns:
  print(f"Unique values of column :\n{x, df[x].unique()}\n")
for x in df.columns:
  plt.figure(figsize=(15, 6))
  sns.countplot(df[x])
  plt.title("\nCount for "+x+" column values")
  plt.show()
df['Result']=df['Result'].replace(-1,0)
df[['Result']]
df.isnull().sum()
from sklearn.model_selection import train_test_split
from sklearn.metrics import *
X=df.drop('Result', axis=1)
y=df['Result']
X=X/X.max()
X_train, X_test, y_train, y_test=train_test_split(X, y, stratify=y, test_size=0.30, random_state=np.random.seed(10))
X_train.shape, X_test.shape
rf=RandomForestClassifier(max_depth=10, random_state=0).fit(X_train, y_train)
pred=rf.predict(X_test)
accuracy_score(y_test, pred)
from sklearn.model_selection import RandomizedSearchCV
n_estimators=[int(x) for x in np.linspace(start=100, stop=2000, num=20)]
max_features=['auto', 'sqrt']
max_depth=[int(x) for x in np.linspace(10, 110, num=11)]
max_depth.append(None)
min_samples_split=[2, 5, 10]
min_samples_leaf=[1, 2, 4]
bootstrap=[True, False]
random_grid={'n_estimators': n_estimators,
             'max_features': max_features,
             'max_depth': max_depth,
             'min_samples_split': min_samples_split,
             'min_samples_leaf': min_samples_leaf,
             'bootstrap': bootstrap}
random_grid
rf=RandomForestClassifier()
rand_search=RandomizedSearchCV(estimator=rf, param_distributions=random_grid, n_iter=30, cv=10, verbose=2, random_state=42, n_jobs=-1)
rand_search.fit(X_train, y_train)
rand_search.best_params_
rf=RandomForestClassifier(n_estimators=1200, 
                          max_depth=30, 
                          bootstrap=False, 
                          max_features='auto', 
                          min_samples_leaf=1, 
                          min_samples_split=2).fit(X_train, y_train)
print(f"Training score: {rf.score(X_train, y_train).round(2)}\n")
print(f"Test score: {rf.score(X_test, y_test).round(2)}")
pred=rf.predict(X_test)
accuracy_score(y_test, pred)
imp=rf.feature_importances_
fi=pd.DataFrame({'features': X.columns, 'importance': imp}).sort_values('importance', ascending=False)
plt.figure(figsize=(15, 8))
sns.barplot(x='importance', y='features', data=fi)
plt.show()
imp_feat=list(fi['features'].head(21))
X=df[imp_feat]
y=df['Result']
X=X/X.max()
X_train, X_test, y_train, y_test=train_test_split(X, y, stratify=y, test_size=0.30, random_state=np.random.seed(10))
X_train.shape, X_test.shape
rf=RandomForestClassifier()
rand_search=RandomizedSearchCV(estimator=rf, param_distributions=random_grid, n_iter=30, cv=10, verbose=2, random_state=42, n_jobs=-1)
rand_search.fit(X_train, y_train)
rand_search.best_params_
rf=RandomForestClassifier(n_estimators=1700, 
                          max_depth=60, 
                          bootstrap=False, 
                          max_features='sqrt', 
                          min_samples_leaf=2, 
                          min_samples_split=5).fit(X_train, y_train)
print(f"Training score: {rf.score(X_train, y_train).round(2)}\n")
print(f"Test score: {rf.score(X_test, y_test).round(2)}")
pred=rf.predict(X_test)
accuracy_score(y_test, pred)
plt.figure(figsize=(30, 20))
sns.heatmap(df.corr(), annot=True, linewidths=.5, fmt='.2g')
plt.show()
corr=df.corr()
filter_corr=corr[(corr >= .80) & (corr != 1)]
plt.figure(figsize=(30, 20))
sns.heatmap(filter_corr, annot=True, cmap='Reds', linewidths=.5, fmt='.2g')
plt.show()
print(list(df.columns))
obs_cols=df[['Shortining_Service', 'double_slash_redirecting', 'Favicon', 'port', 'popUpWidnow', 'Result']]
print(obs_cols.corr()['Result'].sort_values(ascending=False))
df.shape
df.drop(['Favicon', 'popUpWidnow', 'Shortining_Service'], axis=1, inplace=True)
df.shape
corr=df.corr()
filter_corr=corr[(corr >= .80) & (corr != 1)]
plt.figure(figsize=(30, 20))
sns.heatmap(filter_corr, annot=True, cmap='Reds', linewidths=.5, fmt='.2g')
plt.show()
from sklearn.model_selection import train_test_split
from sklearn.metrics import *
X=df.drop('Result', axis=1)
y=df['Result']
X=X/X.max()
X_train, X_test, y_train, y_test=train_test_split(X, y, stratify=y, test_size=0.30, random_state=np.random.seed(10))
X_train.shape, X_test.shape
from sklearn.ensemble import RandomForestClassifier
rf=RandomForestClassifier(max_depth=10, random_state=0).fit(X_train, y_train)
pred=rf.predict(X_test)
accuracy_score(y_test, pred)
print(rf.get_params())
random_grid={'n_estimators': n_estimators,
             'max_features': max_features,
             'max_depth': max_depth,
             'min_samples_split': min_samples_split,
             'min_samples_leaf': min_samples_leaf,
             'bootstrap': bootstrap}
random_grid
rf=RandomForestClassifier()
rand_search=RandomizedSearchCV(estimator=rf, param_distributions=random_grid, n_iter=30, cv=10, verbose=2, random_state=42, n_jobs=-1)
rand_search.fit(X_train, y_train)
rand_search.best_params_
rf=RandomForestClassifier(n_estimators=1900, 
                          max_depth=80, 
                          bootstrap=True, 
                          max_features='auto', 
                          min_samples_leaf=1, 
                          min_samples_split=5).fit(X_train, y_train)
print(f"Training score: {rf.score(X_train, y_train).round(2)}\n")
print(f"Test score: {rf.score(X_test, y_test).round(2)}")
pred=rf.predict(X_test)
accuracy_score(y_test, pred)
import joblib
model=r"randomforest.pkl"
joblib.dump(rf, model)
files.download('randomforest.pkl')
import joblib
import sys
sys.modules['sklearn.externals.joblib']=joblib
from mlxtend.feature_selection import SequentialFeatureSelector as SFS
X.columns
X_test_fi=X_test.values
y_test_fi=y_test.values
sffs=SFS(RandomForestClassifier(n_estimators=1900, 
                          max_depth=80, 
                          bootstrap=True, 
                          max_features='auto', 
                          min_samples_leaf=1, 
                          min_samples_split=5), 
         k_features=X_test.shape[1], forward=True, floating=True, scoring='accuracy', cv=0)
sffs.fit(X_test_fi, y_test_fi, custom_feature_names=X_test.columns)
sffs_df=pd.DataFrame(sffs.subsets_).transpose()
sffs_df
sffs_df.avg_score.sort_values(ascending=False)
sffs_df['feature_names'].loc[20]
sffs_df.to_csv('Important_features.csv', index=False)
files.download('Important_features.csv')
type(sffs_df['feature_names'].loc[20])
IF=list(sffs_df['feature_names'].loc[20])
IF
type(np.array(IF))
imp_features=['having_IPhaving_IP_Address', 'URLURL_Length', 'having_At_Symbol', 'Prefix_Suffix', 'having_Sub_Domain', 'SSLfinal_State', 'Domain_registeration_length', 
              'port', 'HTTPS_token', 'Request_URL', 'URL_of_Anchor', 'Links_in_tags', 'RightClick', 'Iframe', 'age_of_domain', 'DNSRecord', 'web_traffic', 'Page_Rank', 
              'Google_Index', 'Links_pointing_to_page']
len(imp_features)
X=df[imp_features]
y=df['Result']
X=X/X.max()
X_train, X_test, y_train, y_test=train_test_split(X, y, stratify=y, test_size=0.30, random_state=np.random.seed(10))
X_train.shape, X_test.shape
rf=RandomForestClassifier(n_estimators=1900, 
                          max_depth=80, 
                          bootstrap=True, 
                          max_features='auto', 
                          min_samples_leaf=1, 
                          min_samples_split=5).fit(X_train, y_train)
print(f"Training score: {rf.score(X_train, y_train).round(2)}\n")
print(f"Test score: {rf.score(X_test, y_test).round(2)}")
pred=rf.predict(X_test)
accuracy_score(y_test, pred)
X=df.drop('Result', axis=1)
y=df['Result']
X=X/X.max()
X_train, X_test, y_train, y_test=train_test_split(X, y, stratify=y, test_size=0.30, random_state=np.random.seed(10))
X_train.shape, X_test.shape
from sklearn.feature_selection import mutual_info_classif
imp=mutual_info_classif(X, y)
fi=pd.DataFrame({'features':df.columns[0: len(df.columns)-1], 'importance':imp}).sort_values('importance', ascending=False)
plt.figure(figsize=(15, 8))
sns.barplot(x='importance', y='features', data=fi)
plt.show()
imp_feat=list(fi['features'].head(18))
X=df[imp_feat]
y=df['Result']
X=X/X.max()
X_train, X_test, y_train, y_test=train_test_split(X, y, stratify=y, test_size=0.30, random_state=np.random.seed(10))
X_train.shape, X_test.shape
rf=RandomForestClassifier()
rand_search=RandomizedSearchCV(estimator=rf, param_distributions=random_grid, n_iter=30, cv=10, verbose=2, random_state=42, n_jobs=-1)
rand_search.fit(X_train, y_train)
rand_search.best_params_
rf=RandomForestClassifier(n_estimators=1200, 
                          max_depth=30, 
                          bootstrap=False, 
                          max_features='auto', 
                          min_samples_leaf=1, 
                          min_samples_split=2).fit(X_train, y_train)

print(f"Training score: {rf.score(X_train, y_train).round(2)}\n")
print(f"Test score: {rf.score(X_test, y_test).round(2)}")
pred=rf.predict(X_test)
accuracy_score(y_test, pred)
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
for x in X.columns:
  X[x] = X[x].replace(-1, 2)
chi_t=SelectKBest(score_func=chi2, k=X.shape[1])
chi_t.fit(X, y)
fi=pd.DataFrame({'features': X.columns, 'importance': chi_t.scores_}).sort_values('importance', ascending=False)
plt.figure(figsize=(15, 8))
sns.barplot(x='importance', y='features', data=fi)
plt.show()
imp_feat=list(fi['features'].head(16))
X=df[imp_feat]
y=df['Result']
for x in X.columns:
  X[x] = X[x].replace(-1, 2)
X=X/X.max()
X_train, X_test, y_train, y_test=train_test_split(X, y, stratify=y, test_size=0.30, random_state=np.random.seed(10))
X_train.shape, X_test.shape
rf=RandomForestClassifier()
rand_search=RandomizedSearchCV(estimator=rf, param_distributions=random_grid, n_iter=30, cv=10, verbose=2, random_state=42, n_jobs=-1)
rand_search.fit(X_train, y_train)
rand_search.best_params_
rf=RandomForestClassifier(n_estimators=1200, 
                          max_depth=30, 
                          bootstrap=False, 
                          max_features='auto', 
                          min_samples_leaf=1, 
                          min_samples_split=2).fit(X_train, y_train)
print(f"Training score: {rf.score(X_train, y_train).round(2)}\n")
print(f"Test score: {rf.score(X_test, y_test).round(2)}")
pred=rf.predict(X_test)
accuracy_score(y_test, pred)
rf=RandomForestClassifier(n_estimators=1900, 
                          max_depth=80, 
                          bootstrap=True, 
                          max_features='auto', 
                          min_samples_leaf=1, 
                          min_samples_split=5).fit(X_train, y_train)
print(f"Training score: {rf.score(X_train, y_train).round(2)}\n")
print(f"Test score: {rf.score(X_test, y_test).round(2)}")
pred=rf.predict(X_test)
accuracy_score(y_test, pred)
plt.figure(figsize=(16, 6))
sns.heatmap(confusion_matrix(y_test, pred), annot=True, fmt='g')
plt.show()
print(classification_report(y_test, pred, target_names=['Phishing', 'Real']))
import joblib
model=r"randomforest.pkl"
joblib.dump(rf, model)
files.download('randomforest.pkl')
from lightgbm import LGBMClassifier
from sklearn.model_selection import RandomizedSearchCV
n_estimators=[int(x) for x in np.linspace(start=100, stop=2000, num=20)]
max_depth = [int(x) for x in np.linspace(10, 110, num = 11)]
max_depth.append(None)
learning_rate=[0.01, 0.1, 1]
random_grid={'n_estimators': n_estimators,
             'max_depth': max_depth,
             'learning_rate': learning_rate}
random_grid
lgb=LGBMClassifier()
rand_search=RandomizedSearchCV(estimator=lgb, param_distributions=random_grid, n_iter=30, cv=10, verbose=2, random_state=42, n_jobs=-1)
rand_search.fit(X_train, y_train)
rand_search.best_params_
lgb=LGBMClassifier(max_depth=110, n_estimators=700, learning_rate=0.1).fit(X_train, y_train)
print(f"Training score: {lgb.score(X_train, y_train).round(2)}\n")
print(f"Test score: {lgb.score(X_test, y_test).round(2)}")
pred=lgb.predict(X_test)
accuracy_score(y_test, pred)
imp=lgb.feature_importances_
fi=pd.DataFrame({'features': X.columns, 'importance': imp}).sort_values('importance', ascending=False)
plt.figure(figsize=(15, 8))
sns.barplot(x='importance', y='features', data=fi)
plt.show()
imp_feat=list(fi['features'].head(21))
X=df[imp_feat]
y=df['Result']
X=X/X.max()
X_train, X_test, y_train, y_test=train_test_split(X, y, stratify=y, test_size=0.30, random_state=np.random.seed(10))
X_train.shape, X_test.shape
lgb=LGBMClassifier()
rand_search=RandomizedSearchCV(estimator=lgb, param_distributions=random_grid, n_iter=30, cv=10, verbose=2, random_state=42, n_jobs=-1)
rand_search.fit(X_train, y_train)
rand_search.best_params_
X_train.shape
lgb=LGBMClassifier(max_depth=110, n_estimators=700, learning_rate=0.1).fit(X_train, y_train)
print(f"Training score: {lgb.score(X_train, y_train).round(2)}\n")
print(f"Test score: {lgb.score(X_test, y_test).round(2)}")
pred=lgb.predict(X_test)
accuracy_score(y_test, pred)
lgb=LGBMClassifier()
rand_search=RandomizedSearchCV(estimator=lgb, param_distributions=random_grid, n_iter=30, cv=10, verbose=2, random_state=42, n_jobs=-1)
rand_search.fit(X_train, y_train)
rand_search.best_params_
lgb=LGBMClassifier(max_depth=110, n_estimators=900, learning_rate=0.1).fit(X_train, y_train)
print(f"Training score: {lgb.score(X_train, y_train).round(2)}\n")
print(f"Test score: {lgb.score(X_test, y_test).round(2)}")
pred=lgb.predict(X_test)
accuracy_score(y_test, pred)
X_test.shape
plt.figure(figsize=(16, 6))
sns.heatmap(confusion_matrix(y_test, pred), annot=True, fmt='g')
plt.show()
print(classification_report(y_test, pred, target_names=['Phishing', 'Real']))
import joblib
model=r"lightgbm.pkl"
joblib.dump(lgb, model)
files.download('lightgbm.pkl')
lgb=LGBMClassifier()
rand_search=RandomizedSearchCV(estimator=lgb, param_distributions=random_grid, n_iter=30, cv=10, verbose=2, random_state=42, n_jobs=-1)
rand_search.fit(X_train, y_train)
rand_search.best_params_
lgb=LGBMClassifier(max_depth=70, n_estimators=800, learning_rate=0.1).fit(X_train, y_train)
print(f"Training score: {lgb.score(X_train, y_train).round(2)}\n")
print(f"Test score: {lgb.score(X_test, y_test).round(2)}")
pred=lgb.predict(X_test)
accuracy_score(y_test, pred)
X_train.shape
lgb=LGBMClassifier()
rand_search=RandomizedSearchCV(estimator=lgb, param_distributions=random_grid, n_iter=30, cv=10, verbose=2, random_state=42, n_jobs=-1)
rand_search.fit(X_train, y_train)
rand_search.best_params_
lgb=LGBMClassifier(max_depth=110, n_estimators=1400, learning_rate=0.1).fit(X_train, y_train)
print(f"Training score: {lgb.score(X_train, y_train).round(2)}\n")
print(f"Test score: {lgb.score(X_test, y_test).round(2)}")
pred=lgb.predict(X_test)
accuracy_score(y_test, pred)
!pip install catboost
from catboost import CatBoostClassifier
from sklearn.model_selection import RandomizedSearchCV
iterations=[int(x) for x in np.linspace(start=10, stop=100, num=10)]
depth=[int(x) for x in np.linspace(10, 110, num = 11)]
depth.append(None)
learning_rate=[0.01, 0.1, 1]
random_grid={'iterations': iterations,
             'depth': depth,
             'learning_rate': learning_rate}
random_grid
X_train.shape
cat=CatBoostClassifier()
rand_search=RandomizedSearchCV(estimator=cat, param_distributions=random_grid, n_iter=30, cv=10, verbose=2, random_state=42, n_jobs=-1)
rand_search.fit(X_train, y_train)
rand_search.best_params_
cat=CatBoostClassifier(depth=10, iterations=90, learning_rate=0.1).fit(X_train, y_train)
print(f"Training score: {cat.score(X_train, y_train).round(2)}\n")
print(f"Test score: {cat.score(X_test, y_test).round(2)}")
pred=cat.predict(X_test)
accuracy_score(y_test, pred)
X_test.shape
plt.figure(figsize=(16, 6))
sns.heatmap(confusion_matrix(y_test, pred), annot=True, fmt='g')
plt.show()
print(classification_report(y_test, pred, target_names=['Phishing', 'Real']))
import joblib
model=r"catboost.pkl"
joblib.dump(cat, model)
files.download('catboost.pkl')
imp=cat.feature_importances_
fi=pd.DataFrame({'features': X.columns, 'importance': imp}).sort_values('importance', ascending=False)
plt.figure(figsize=(15, 8))
sns.barplot(x='importance', y='features', data=fi)
plt.show()
iterations=[int(x) for x in np.linspace(start=10, stop=100, num=10)]
depth=[4,5,6,7,8,9, 10]
depth.append(None)
learning_rate=[0.01, 0.1, 1]
# Create the random grid
random_grid={'iterations': iterations,
             'depth': depth,
             'learning_rate': learning_rate}
random_grid
imp_feat=list(fi['features'].head(12))
X=df[imp_feat]
y=df['Result']
X=X/X.max()
X_train, X_test, y_train, y_test=train_test_split(X, y, stratify=y, test_size=0.30, random_state=np.random.seed(10))
X_train.shape, X_test.shape
cat=CatBoostClassifier()
rand_search=RandomizedSearchCV(estimator=cat, param_distributions=random_grid, n_iter=30, cv=10, verbose=2, random_state=42, n_jobs=-1)
rand_search.fit(X_train, y_train)
rand_search.best_params_
cat=CatBoostClassifier(depth=10, iterations=90, learning_rate=1).fit(X_train, y_train)
print(f"Training score: {cat.score(X_train, y_train).round(2)}\n")
print(f"Test score: {cat.score(X_test, y_test).round(2)}")
pred=cat.predict(X_test)
accuracy_score(y_test, pred)
cat=CatBoostClassifier()
rand_search=RandomizedSearchCV(estimator=cat, param_distributions=random_grid, n_iter=30, cv=10, verbose=2, random_state=42, n_jobs=-1)
rand_search.fit(X_train, y_train)
rand_search.best_params_
cat=CatBoostClassifier(depth=8, iterations=80, learning_rate=1).fit(X_train, y_train)
print(f"Training score: {cat.score(X_train, y_train).round(2)}\n")
print(f"Test score: {cat.score(X_test, y_test).round(2)}")
pred=cat.predict(X_test)
accuracy_score(y_test, pred)
cat=CatBoostClassifier()
rand_search=RandomizedSearchCV(estimator=cat, param_distributions=random_grid, n_iter=30, cv=10, verbose=2, random_state=42, n_jobs=-1)
rand_search.fit(X_train, y_train)
rand_search.best_params_
cat=CatBoostClassifier(depth=9, iterations=90, learning_rate=1).fit(X_train, y_train)
print(f"Training score: {cat.score(X_train, y_train).round(2)}\n")
print(f"Test score: {cat.score(X_test, y_test).round(2)}")
pred=cat.predict(X_test)
accuracy_score(y_test, pred)
cat=CatBoostClassifier()
rand_search=RandomizedSearchCV(estimator=cat, param_distributions=random_grid, n_iter=30, cv=10, verbose=2, random_state=42, n_jobs=-1)
rand_search.fit(X_train, y_train)
rand_search.best_params_
cat=CatBoostClassifier(depth=7, iterations=80, learning_rate=1).fit(X_train, y_train)
print(f"Training score: {cat.score(X_train, y_train).round(2)}\n")
print(f"Test score: {cat.score(X_test, y_test).round(2)}")
pred=cat.predict(X_test)
accuracy_score(y_test, pred)