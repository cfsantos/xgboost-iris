from sklearn import datasets
from xgboost import XGBClassifier
from xgboost import plot_importance
from matplotlib import pyplot

# Carrega automaticamente o dataset
iris = datasets.load_iris()
X = iris.data[:, :]  
y = iris.target

# classifica os dados
model = XGBClassifier()
model.fit(X, y)

# Gera grafico com importancias
plot_importance(model)
pyplot.show()
