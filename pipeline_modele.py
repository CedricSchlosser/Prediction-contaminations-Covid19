# -*- coding: utf-8 -*-

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor #il faut l'installer sur votre environnement, ouvrir le cmd + "pip install xgboost"
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import RepeatedKFold, cross_val_score
df = pd.read_csv("Donnee_Covid_Traitees.csv")
df.info()
print("On regarde le df",df.head())
"""
On constate des colonnes colinéaires : Saison été / Saison hiver, Convifinement Oui / Confinement Non"
On choisi de les droper
"""

df = df.drop(["saison_hiver", "confinement_non"], axis = 1)

"""
Il faut déterminer notre variable cible
La période moyenne d'incubation du covid 19 est de 5 jours
Nous partons de l'hypothèse qu'à date T+0, les contaminés seront déclarés positifs à T+5 
C'est une grosse simplification, car les temps d'incubation sont variables, et toutes les personnes ne se font pas tester
Il semble cependant que cela soit le plus raisonnable pour nous

La variable cible est donc : 
le nombre cas positif à T+5

Nous créons une colonne "target" qui correspond à celà
"""

df_cible = df.loc[5:,["pos"]]
#On renomme la colonne
df_cible = df_cible.rename(columns={"pos" : "pos+5"})
#On supprime les 5 dernière ligne du df de base, car il n'y a pas de variable cible à mettre en face
df = df.iloc[:-5, :]


#On transforme nos dates en ordonnal
df["date"] = pd.to_datetime(df["date"], format = '%Y/%m/%d')
df["date"] = df['date'].apply(datetime.toordinal)


#On scale les données de nos features sans la colonne date
df_scale = df.drop(["date"], axis = 1)
scaler = StandardScaler()
df_scaled = scaler.fit_transform(df_scale)
df_scaled = pd.DataFrame(data = df_scaled)
#on remet les noms de colonnes
df_scaled.columns = df_scale.columns
#on remet notre date
df_scaled["date"] = df["date"]

#On plot une matrice de corrélation pour détecter (ou non) la présence de corrélations trop élevées nécéssitant éventuellement la suppression de certaines données
cor = df_scaled.corr()
plt.figure(figsize=(15,15))
sns.heatmap(cor, square = True, cmap="coolwarm",annot=True)
#Présence de corrélations très proches de 1 sur certaines variables. Constat à garder en tête pour la suite.





#On fait notre train/test
#X_train, X_test, y_train, y_test = train_test_split(df_scaled, df_cible, test_size = 0.2, random_state = 124)
#On n'utilise pas la fonction classique, on va supposer que le temps a un impact, donc on garde l'ordre
X_train = df_scaled[0:344]
X_test = df_scaled[344:]
y_train = df_cible[0:344]
y_test = df_cible[344:]
#Modèle "basique" de randomforestregressor, on va chercher les variables les plus importantes
model = RandomForestRegressor(random_state = 52)
model = model.fit(X_train, y_train.values.ravel())
# On cherche à savoir quel sont les variables les importantes pour notre modèle prédictif
importances = model.feature_importances_
varRandomForest = []
std = np.std([tree.feature_importances_ for tree in model.estimators_], axis=0)
forest_importances = pd.Series(importances, index=df_scaled.columns)
forest_importances = forest_importances.sort_values(ascending = False)

#On exécute ce bloc tout seul
fig, ax = plt.subplots()
forest_importances.plot.bar(yerr=std, ax=ax)
ax.set_title("Importance de nos features dans le modèle")
ax.set_ylabel("Mean decrease in impurity")
fig.tight_layout()

#Puis ce bloc
data = pd.DataFrame(forest_importances[:20]).transpose()
ax = sns.barplot(data=data, orient = 'h')
ax.set(xlabel='Importance feature avec MDI', ylabel='Features')
#On constate que la date est nulle et la variable pos n'est pas très élevé non plus


#On choisi de garder seulement les 9 varoables les plus importantes
varRandomForest = list(forest_importances[0:9].index)
df_scaled_opti_RFR = pd.DataFrame()
for i in varRandomForest:
    df_scaled_opti_RFR[i] = df_scaled[i]


#On refait le même modèle mais avec nos nouvelles variables
X_train = df_scaled_opti_RFR[0:344]
X_test = df_scaled_opti_RFR[344:]
model_var_opti_RFR = RandomForestRegressor(random_state = 52)
model_var_opti_RFR = model_var_opti_RFR.fit(X_train, y_train.values.ravel())
importances = model_var_opti_RFR.feature_importances_
std = np.std([tree.feature_importances_ for tree in model_var_opti_RFR.estimators_], axis=0)

forest_importances = pd.Series(importances, index=df_scaled_opti_RFR.columns)
forest_importances = forest_importances.sort_values(ascending = False)

fig, ax = plt.subplots()
forest_importances.plot.bar(yerr=std, ax=ax)
ax.set_title("Importance de nos features dans le modèle opti")
ax.set_ylabel("Mean decrease in impurity")
fig.tight_layout()

data = pd.DataFrame(forest_importances[:9]).transpose()
ax = sns.barplot(data=data, orient = 'h')
ax.set(xlabel='Importance feature avec MDI', ylabel='Features')




#On refait la même chose avec XGBoost
X_train = df_scaled[0:344]
X_test = df_scaled[344:]
model_XG = XGBRegressor(seed = 52)
model_XG = model_XG.fit(X_train, y_train)

importances = model.feature_importances_
varXGB = []
XGB_importances = pd.Series(importances, index=df_scaled.columns)
XGB_importances = XGB_importances.sort_values(ascending = False)

#on exécute ce premier bloc
fig, ax = plt.subplots()
XGB_importances.plot.bar( ax=ax)
ax.set_title("Feature importances using MDI")
ax.set_ylabel("Mean decrease in impurity")
fig.tight_layout()
#Constat : nos premières variables sont les mêmes

data = pd.DataFrame(XGB_importances[:20]).transpose()
ax = sns.barplot(data=data, orient = 'h')
ax.set(xlabel='Importance feature avec MDI', ylabel='Features')

#On garde les neufs premières variables
varXGB = list(XGB_importances[0:9].index)
df_scaled_opti_XGB = pd.DataFrame()
for i in varXGB:
    df_scaled_opti_XGB[i] = df_scaled[i]
    
#On refait le modèle
X_train = df_scaled_opti_XGB[0:344]
X_test = df_scaled_opti_XGB[344:]
model_opti_XGB = XGBRegressor(seed = 52)
model_opti_XGB = model_opti_XGB.fit(X_train, y_train)

importances = model_opti_XGB.feature_importances_
XGB_importances = pd.Series(importances, index=df_scaled_opti_XGB.columns)
XGB_importances = XGB_importances.sort_values(ascending = False)

fig, ax = plt.subplots()
XGB_importances.plot.bar( ax=ax)
ax.set_title("Feature importances using MDI")
ax.set_ylabel("Mean decrease in impurity")
fig.tight_layout()

data = pd.DataFrame(XGB_importances[:9]).transpose()
ax = sns.barplot(data=data, orient = 'h')
ax.set(xlabel='Importance feature avec MDI', ylabel='Features')

#On constate ici que tx_incid n'a plus d'importance, on choisi donc de le dropper
varXGB = list(XGB_importances[0:8].index)
df_scaled_opti_XGB = pd.DataFrame()
for i in varXGB:
    df_scaled_opti_XGB[i] = df_scaled[i]

#On optimise les hyperparamètres
#On commence par RFR
X_train_opti_param_RFR = df_scaled_opti_RFR[0:344]
X_test_opti_param_RFR= df_scaled_opti_RFR[344:]
# Nombre  d'arbre
n_estimators = [int(x) for x in np.linspace(start = 200, stop = 2000, num = 10)]
# Nb de FT à prendre en compte à chaque feuille
max_features = ['auto', 'sqrt']
# Profondeur max
max_depth = [int(x) for x in np.linspace(10, 110, num = 11)]
max_depth.append(None)
# Combien de sample pour split au minimum
min_samples_split = [2, 5, 10]
# Combien de sample mini pour chaque feuille
min_samples_leaf = [1, 2, 4]
# Methode d'entrainement
bootstrap = [True, False]
# Create the random grid
random_grid = {'n_estimators': n_estimators,
               'max_features': max_features,
               'max_depth': max_depth,
               'min_samples_split': min_samples_split,
               'min_samples_leaf': min_samples_leaf,
               'bootstrap': bootstrap}
print(random_grid)

# Mèdle de base
rf = RandomForestRegressor()
# Recherche aléatoire avec CV =3
# 100 iter, on utilise tous nos coeurs
rf_random = RandomizedSearchCV(estimator = rf, param_distributions = random_grid, n_iter = 100, cv = 3, verbose=2, random_state=52, n_jobs = -1)
# Fit the random search model
res_cv_rf = rf_random.fit(X_train_opti_param_RFR, y_train.values.ravel())
res_cv_rf.best_estimator_

# Optimisation XGB
X_train_opti_param_XGB = df_scaled_opti_XGB[0:344]
X_test_opti_param_XGB = df_scaled_opti_XGB[344:]
params_XGB = {
    # Parameters à optimiser.
    'max_depth':6,
    'min_child_weight': 1,
    'eta':.3,
    'subsample': 1,
    'colsample_bytree': 1,
    #Autre
    'objective':'reg:squarederror'}

import xgboost as xgb
dtrain = xgb.DMatrix(X_train_opti_param_XGB,label=y_train)
dtest = xgb.DMatrix(X_test_opti_param_XGB,label=y_test)
num_boost_round = 10

gridsearch_params = [
    (max_depth, min_child_weight)
    for max_depth in range(5,12)
    for min_child_weight in range(5,12)]

min_mae = float("Inf")
best_params = None
for max_depth, min_child_weight in gridsearch_params:
    print("CV with max_depth={}, min_child_weight={}".format(
                             max_depth,
                             min_child_weight))
    # Update les parameters
    params_XGB['max_depth'] = max_depth
    params_XGB['min_child_weight'] = min_child_weight
    # Run CV
    cv_results = xgb.cv(
        params_XGB,
        dtrain,
        num_boost_round=num_boost_round,
        seed=42,
        nfold=5,
        metrics={'mae'},
        early_stopping_rounds=10
    )
    # Update best MAE
    mean_mae = cv_results['test-mae-mean'].min()
    boost_rounds = cv_results['test-mae-mean'].argmin()
    print("\t mae {} for {} rounds".format(mean_mae, boost_rounds))
    if mean_mae < min_mae:
        min_mae = mean_mae
        best_params = (max_depth,min_child_weight)
print("Best params: {}, {}, mae: {}".format(best_params[0], best_params[1], min_mae))
params_XGB['max_depth'] = best_params[0]
params_XGB['min_child_weight'] = best_params[1]

#On optimise les autres paramètre du XGBoost
gridsearch_params = [
    (subsample, colsample)
    for subsample in [i/10. for i in range(3,11)]
    for colsample in [i/10. for i in range(3,11)]
]

min_mae = float("Inf")
best_params = None
# On démarre par la plus grosse valeur pour terminer à la plus petite
for subsample, colsample in reversed(gridsearch_params):
    print("CV with subsample={}, colsample={}".format(
                             subsample,
                             colsample))
    # On update le param
    params_XGB['subsample'] = subsample
    params_XGB['colsample_bytree'] = colsample
    # On fait tourner le CV
    cv_results = xgb.cv(
        params_XGB,
        dtrain,
        num_boost_round=num_boost_round,
        seed=42,
        nfold=5,
        metrics={'mae'}, #On change ici la metric
        early_stopping_rounds=10
    )
    # Update la meilleure mae
    mean_mae = cv_results['test-mae-mean'].min()
    boost_rounds = cv_results['test-mae-mean'].argmin()
    print("\t mae {} for {} rounds".format(mean_mae, boost_rounds))
    if mean_mae < min_mae:
        min_mae = mean_mae
        best_params = (subsample,colsample)
print("Best params: {}, {}, mae: {}".format(best_params[0], best_params[1], min_mae))
params_XGB['subsample'] = best_params[0]
params_XGB['colsample_bytree'] = best_params[1]

#dernière opti XGBoost
min_mae = float("Inf")
best_params = None
for eta in [.5, .4, .3, .2, .1, .05, .01, .005]:
    print("CV with eta={}".format(eta))
    # Update des parametres
    params_XGB['eta'] = eta
    # Run and time CV
    cv_results = xgb.cv(
            params_XGB,
            dtrain,
            num_boost_round=num_boost_round,
            seed=42,
            nfold=5,
            metrics=['mae'],
            early_stopping_rounds=10
          )
    # Update des meilleurs scores
    mean_mae = cv_results['test-mae-mean'].min()
    boost_rounds = cv_results['test-mae-mean'].argmin()
    print("\tMAE {} for {} rounds\n".format(mean_mae, boost_rounds))
    if mean_mae < min_mae:
        min_mae = mean_mae
        best_params = eta
print("Best params: {}, MAE: {}".format(best_params, min_mae))
params_XGB['eta'] = best_params
print("Les meilleurs params XGB sont : ", params_XGB)


#TEST XGB 
XGB_final = XGBRegressor(max_depth = params_XGB['max_depth'],
                               min_child_weight = params_XGB['min_child_weight'],
                               eta = params_XGB['eta'],
                               subsample = params_XGB['subsample'],
                               colsample_bytree = params_XGB['colsample_bytree'],
                               objective = params_XGB['objective'],
                               seed = 52)
cv = RepeatedKFold(n_splits=10, n_repeats=3, random_state=1)
# Evaluation du modèle
scores = cross_val_score(XGB_final, df_scaled_opti_XGB, df_cible, scoring='neg_mean_absolute_error', cv=cv, n_jobs=-1)
# Scores positifs
scores = np.absolute(scores)
print('Mean MAE: %.3f (%.3f)' % (scores.mean(), scores.std()) )

XGB_final.fit(X_train_opti_param_XGB, y_train)
y_XGB_pred = XGB_final.predict(X_test_opti_param_XGB)
y_XGB_pred = pd.DataFrame(y_XGB_pred)
y_test = df_cible[344:]
y_test = y_test.reset_index()
y_test = y_test.drop("index", axis = 1)
plt.plot(y_XGB_pred, label = "Prédictions XGB", color = "b")
plt.plot(y_test, label = "Test", color = "red")
plt.legend()
plt.title("XGB VS Test avec samedi et dimanche", color="#0077BE")
# Constat : notre Modèle capte complètement la tendance. 
# Remarque : On voit sur les données de test des gros pics négatifs : ce sont les dimanche & les samedi. Des jours ou aucunes données ne sont enregistrées
# On choisit de les supprimer
y_test = df_cible[344:]
y_test = y_test.reset_index()
samedi = np.arange(5, 86, 7)
dimanche = np.arange(6, 86, 7)
for i in samedi:
    y_test = y_test.drop(i)
    y_XGB_pred = y_XGB_pred.drop(i)
for i in dimanche:
    y_test = y_test.drop(i)
    y_XGB_pred = y_XGB_pred.drop(i)

y_test = y_test.drop("index", axis = 1)
plt.plot(y_XGB_pred, color = "b", label = "XGB")
plt.plot(y_test, color = "r", label = "Test")
plt.title("XGB VS Test sans les dimanche et samedi")
plt.legend()
# Remarques :
# Il reste encore des valeurs "abérantes" correspondant probablement aux jours fériés 
legend = plt.legend()
plt.setp(legend.get_texts(), color="#0077BE")
# Alternative proposée 1 : Re-entrainer le modèle sans samedi/dimanche/jours fériés
# Alternative proposée 2 : Re-entrainer le modèle en "flaguant" comme confinement/hiver les jours où l'on ne se teste pas

# Modèle RFR final
y_test = df_cible[344:]
y_test = y_test.reset_index()
RFR_final = RandomForestRegressor(bootstrap = True,
                                  criterion = "mse",
                                  max_depth=40, max_features='sqrt', 
                                  min_samples_split=5,
                                  n_estimators=200)
cv = RepeatedKFold(n_splits=10, n_repeats=3, random_state=1)
# Evaluation du modele
scores = cross_val_score(RFR_final, df_scaled_opti_RFR, df_cible, scoring='neg_mean_absolute_error', cv=cv, n_jobs=-1)
# Scores positifs
scores = np.absolute(scores)
print('Mean MAE: %.3f (%.3f)' % (scores.mean(), scores.std()) )

RFR_final.fit(X_train_opti_param_RFR, y_train)
y_RFR_pred = RFR_final.predict(X_test_opti_param_RFR)
y_RFR_pred = pd.DataFrame(y_RFR_pred)

y_test = y_test.drop("index", axis = 1)

plt.plot(y_RFR_pred, color="b", label = "RFR")
plt.plot(y_test, color="r", label = "test")
plt.title("RFR VS Test avec samedi/dimanche")
plt.legend()
y_test = df_cible[344:]
y_test = y_test.reset_index()
for i in samedi:
    y_test = y_test.drop(i)
    y_RFR_pred = y_RFR_pred.drop(i)
for i in dimanche:
    y_test = y_test.drop(i)
    y_RFR_pred = y_RFR_pred.drop(i)



y_test = y_test.drop("index", axis = 1)
plt.plot(y_RFR_pred, color = "b", label ="RFR")
plt.plot(y_test, color = "r", label ="Test")
plt.legend()
plt.title("RFR VS Test sans samedi dimanche")

from sklearn.metrics import mean_absolute_error
print("MAE finale XGB boost sans samedi/dimanche", mean_absolute_error(y_test, y_XGB_pred))
print("MAE finale RFR sans samedi/dimanche", mean_absolute_error(y_test, y_RFR_pred))
# Constat final : Le modèle XGB offre de meilleurs résultats, il capte bien la tendance des données mais et à une MAE plus faible que RFR