# Proyecto de Data Science

Este proyecto incluye dos partes importantes: **Cross Validation** y **Randomized Search**.

## Cross Validation

En esta sección, utilizamos Cross Validation para evaluar el rendimiento de un modelo de regresión de árbol de decisión en el conjunto de datos `ECommerce_consumer_behaviour.csv`.

```python
        if __name__ == "__main__":
            dataset = pd.read_csv('./data/ECommerce_consumer_behaviour.csv')

            X = dataset.drop(['reordered', 'department', 'product_name', 'days_since_prior_order'], axis=1)
            y = dataset['reordered']

            model = DecisionTreeRegressor()
            score = cross_val_score(model, X, y, cv=3, scoring='neg_mean_squared_error')
            print("="*64)
            print(score)
            print("="*64)
            print(np.abs(np.mean(score)))
            print("="*64)

            kf = KFold(n_splits=3, shuffle=True, random_state=42)
            for train, test in kf.split(dataset):
                print("="*64)
                print(train)
                print("="*64)
                print(test)
                print("="*64)
```
## Randomized Search
En esta sección, aplicamos Randomized Search para optimizar los hiperparámetros de un modelo RandomForestRegressor en el conjunto de datos diamonds.csv.

´´´python
    if __name__ == "__main__":
        dataset = pd.read_csv('./data/diamonds.csv')
        print(dataset)

        X = dataset.drop(['cut', 'color', 'clarity'], axis=1)
        y = dataset['price']

        reg = RandomForestRegressor()

        parametros = {
            'n_estimators': range(4, 16),
            'criterion': ['friedman_mse', 'squared_error', 'poisson', 'absolute_error'],
            'max_depth': range(2, 11)
        }

        rand_est = RandomizedSearchCV(reg, parametros, n_iter=10, cv=3, scoring='neg_mean_absolute_error').fit(X, y)
        
        print("="*64)
        print(rand_est.best_estimator_)
        print("="*64)
        print(rand_est.best_params_)
        print("="*64)
        print(rand_est.predict(X.loc[[0]]))
        print("="*64)
´´´
¡Gracias por revisar nuestro proyecto! Si tienes alguna pregunta, no dudes en contactarnos.