Modellerin ön hazırlık aşamaları ihtiyaca göre değiştirilebilir.

Öznitelik Seçimi yapılırken: SelectKBest kullanarak hedef değişkenle en yüksek ilişkiye sahip 5 özniteliği seçtik.
Sonrasında Model Eğitimi: Sadece en iyi özelliklerle modelleri yeniden eğittik.

# En iyi özellikleri seçme
selector = SelectKBest(score_func=f_regression, k=5)
X_selected = selector.fit_transform(X, y)
selected_features = X.columns[selector.get_support()]
print("Seçilen Öznitelikler:", selected_features)

Başka teknikler geliştirmek istersek:
1. Mutual Information (Karşılıklı Bilgi)
```python
from sklearn.feature_selection import mutual_info_regression

selector = SelectKBest(score_func=mutual_info_regression, k=5)
X_selected_mi = selector.fit_transform(X, y)
selected_features_mi = X.columns[selector.get_support()]
print("Mutual Information Seçilen Öznitelikler:", selected_features_mi)
```

2. Recursive Feature Elimination (RFE)
```python
from sklearn.feature_selection import RFE

rfe_selector = RFE(estimator=LinearRegression(), n_features_to_select=5, step=1)
X_selected_rfe = rfe_selector.fit_transform(X, y)
selected_features_rfe = X.columns[rfe_selector.get_support()]
print("RFE Seçilen Öznitelikler:", selected_features_rfe)
```

3. Principal Component Analysis (PCA)
```python
from sklearn.decomposition import PCA

pca = PCA(n_components=5)
X_selected_pca = pca.fit_transform(X)
print("PCA ile seçilen 5 bileşen:", X_selected_pca.shape)
```

4. L1-Based Feature Selection (Lasso)
```python
from sklearn.linear_model import Lasso
from sklearn.feature_selection import SelectFromModel

lasso = Lasso(alpha=0.1)
lasso.fit(X, y)
selector = SelectFromModel(lasso, prefit=True)
X_selected_lasso = selector.transform(X)
selected_features_lasso = X.columns[selector.get_support()]
print("Lasso Seçilen Öznitelikler:", selected_features_lasso)
```

5. Tree-Based Feature Selection
```python
from sklearn.ensemble import RandomForestRegressor

rf = RandomForestRegressor(n_estimators=100, random_state=42)
rf.fit(X, y)
selector = SelectFromModel(rf, prefit=True)
X_selected_rf = selector.transform(X)
selected_features_rf = X.columns[selector.get_support()]
print("Random Forest Seçilen Öznitelikler:", selected_features_rf)
```
