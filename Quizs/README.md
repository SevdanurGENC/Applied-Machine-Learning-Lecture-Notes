Modellerin ön hazırlık aşamaları ihtiyaca göre değiştirilebilir.

Öznitelik Seçimi yapılırken: SelectKBest kullanarak hedef değişkenle en yüksek ilişkiye sahip 5 özniteliği seçtik.
Sonrasında Model Eğitimi: Sadece en iyi özelliklerle modelleri yeniden eğittik.
```python
# En iyi özellikleri seçme
selector = SelectKBest(score_func=f_regression, k=5)
X_selected = selector.fit_transform(X, y)
selected_features = X.columns[selector.get_support()]
print("Seçilen Öznitelikler:", selected_features)
```

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

6. Variance Threshold (Varyans Eşiği)
   ```python
   from sklearn.feature_selection import VarianceThreshold

   selector = VarianceThreshold(threshold=0.1)
   X_selected_variance = selector.fit_transform(X)
   selected_features_variance = X.columns[selector.get_support()]
   print("Variance Threshold Seçilen Öznitelikler:", selected_features_variance)
   ```

7. F-Regression (Klasik İstatistiksel Yöntem)
   ```python
   from sklearn.feature_selection import f_regression

   selector = SelectKBest(score_func=f_regression, k=5)
   X_selected_freg = selector.fit_transform(X, y)
   selected_features_freg = X.columns[selector.get_support()]
   print("F-Regression Seçilen Öznitelikler:", selected_features_freg)
   ```

8. Elastic Net (L1 ve L2 Regülerleştirme Kombinasyonu)
   ```python
   from sklearn.linear_model import ElasticNet
   from sklearn.feature_selection import SelectFromModel

   enet = ElasticNet(alpha=0.1, l1_ratio=0.5)
   enet.fit(X, y)
   selector = SelectFromModel(enet, prefit=True)
   X_selected_enet = selector.transform(X)
   selected_features_enet = X.columns[selector.get_support()]
   print("Elastic Net Seçilen Öznitelikler:", selected_features_enet)
   ```

9. SHAP (Shapley Additive Explanations)
   ```python
   import shap

   model = RandomForestRegressor(n_estimators=100)
   model.fit(X, y)
   explainer = shap.Explainer(model, X)
   shap_values = explainer(X)
   shap.summary_plot(shap_values, X)
   ```

10. Boruta Algorithm (Random Forest ile Seçim)
   ```python
   from boruta import BorutaPy

   rf = RandomForestRegressor(n_jobs=-1)
   boruta = BorutaPy(rf, n_estimators='auto', verbose=2, random_state=42)
   boruta.fit(X.values, y.values)

   selected_features_boruta = X.columns[boruta.support_]
   print("Boruta Seçilen Öznitelikler:", selected_features_boruta)
   ```

Bu öznitelik seçme tekniklerini de lütfen deneyerek sonuçları alınız ve karşılaştırınız (her defasında veri kümesinde tek bir öznitelik seçme tekniği kullanınız, birden fazla kullanılmaz).
