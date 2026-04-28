# ################################################################
# PROJETO FINAL
#
# Universidade Federal de São Carlos (UFSCar)
# Departamento de Computação - Sorocaba (DComp-So)
# Disciplina: Aprendizado de Máquina
# Prof. Tiago A. Almeida
#
# Nome: Anne Mari Suenaga Sakai
# RA: 822304
# 
# Nome: Felipe Jun Nishitani
# RA: 822353
# ################################################################

# Arquivo com todas as funções e códigos referentes aos experimentos

import os
import pandas as pd
import numpy as np

# Modelos
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

# ===================================================================
# Treinamento dos modelos (BÁSICOS / BASELINES)
# ===================================================================

def treinar_knn(X_train, y_train, k=7):
    """
    Treina um modelo KNN simples (baseline).
    """
    print(f"\n--- Treinando KNN (k={k}) ---")
    model = KNeighborsClassifier(n_neighbors=k)
    model.fit(X_train, y_train)
    return model


def treinar_naive_bayes(X_train, y_train):
    """
    Treina Naive Bayes (baseline).
    """
    print("\n--- Treinando Naive Bayes ---")
    model = GaussianNB()
    model.fit(X_train, y_train)
    return model


def treinar_regressao_logistica(X_train, y_train):
    """
    Regressão logística (baseline forte).
    """
    print("\n--- Treinando Regressão Logística ---")
    model = LogisticRegression(
        max_iter=1000,
        solver="lbfgs",
        random_state=42
    )
    model.fit(X_train, y_train)
    return model


def treinar_rede_neural(X_train, y_train):
    """
    Rede neural MLP simples (baseline).
    """
    print("\n--- Treinando Rede Neural (MLP) ---")
    model = MLPClassifier(
        hidden_layer_sizes=(64, 32),
        activation='relu',
        solver='adam',
        max_iter=500,
        random_state=42
    )
    model.fit(X_train, y_train)
    return model


def treinar_svm(X_train, y_train):
    """
    SVM simples (baseline).
    """
    print("\n--- Treinando SVM (RBF) ---")
    model = SVC(
        kernel='rbf',
        probability=True,
        C=1.0,
        gamma='scale',
        random_state=42
    )
    model.fit(X_train, y_train)
    return model


def treinar_random_forest(X_train, y_train):
    """
    Random Forest com parâmetros padrão (baseline).
    """
    print("\n--- Treinando Random Forest (baseline) ---")
    model = RandomForestClassifier(
        n_estimators=100,
        random_state=42,
        n_jobs=1
    )
    model.fit(X_train, y_train)
    return model


def treinar_xgboost(X_train, y_train):
    """
    XGBoost com parâmetros padrão (baseline).
    """
    print("\n--- Treinando XGBoost (baseline) ---")

    model = XGBClassifier(
        n_estimators=200,
        learning_rate=0.05,
        max_depth=5,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="multi:softprob",
        eval_metric="mlogloss",
        random_state=42,
        n_jobs=-1
    )

    model.fit(X_train, y_train)
    return model


# ===================================================================
# AVALIAÇÃO VIA VALIDAÇÃO CRUZADA
# ===================================================================

def avaliar_com_crossval(model, X, y, cv=5, scoring="accuracy"):
    """
    Executa validação cruzada (k-fold).
    Retorna os scores obtidos.
    """
    print(f"\n--- Avaliação com Validação Cruzada ({cv}-fold) ---")
    scores = cross_val_score(model, X, y, cv=cv, scoring=scoring)
    print("Scores individuais:", scores)
    print("Média:", scores.mean())
    print("Desvio:", scores.std())
    return scores



# ===================================================================
# MODELOS OTIMIZADOS COM GRID SEARCH
# ===================================================================

def treinar_svm_otimizado(X_train, y_train):
    """
    Ajuste de hiperparâmetros do SVM usando GridSearchCV.
    """
    print("\n=== Ajustando SVM com GridSearchCV ===")

    param_grid = {
        "C": [0.1, 1, 5, 10, 50],
        "gamma": ["scale", 0.1, 0.01, 0.001],
        "kernel": ["rbf"]
    }

    grid = GridSearchCV(
        estimator=SVC(probability=True, random_state=42),
        param_grid=param_grid,
        cv=5,
        scoring="accuracy",
        n_jobs=-1
    )

    grid.fit(X_train, y_train)

    print("Melhores parâmetros:", grid.best_params_)
    print("Melhor score médio CV:", grid.best_score_)

    return grid.best_estimator_


def treinar_mlp_otimizado(X_train, y_train):
    """
    Ajuste de hiperparâmetros do MLP (rede neural).
    """
    print("\n=== Ajustando MLP com GridSearchCV ===")

    param_grid = {
        "hidden_layer_sizes": [(64,), (128,), (64, 32), (128, 64)],
        "learning_rate_init": [0.001, 0.01],
        "alpha": [0.0001, 0.001],
        "activation": ["relu"]
    }

    grid = GridSearchCV(
        estimator=MLPClassifier(max_iter=600, random_state=42),
        param_grid=param_grid,
        cv=5,
        scoring="accuracy",
        n_jobs=-1
    )

    grid.fit(X_train, y_train)

    print("Melhores parâmetros:", grid.best_params_)
    print("Melhor score CV:", grid.best_score_)

    return grid.best_estimator_


def treinar_rf_otimizado(X_train, y_train):
    """
    Ajuste de hiperparâmetros do Random Forest.
    """
    print("\n=== Ajustando Random Forest com GridSearchCV ===")

    param_grid = {
        'n_estimators': [50, 100, 200],
        'max_depth': [None, 5, 10, 15],
        'criterion': ['gini', 'entropy'],
        'min_samples_split': [5, 10, 15],
        'min_samples_leaf': [2, 4, 6],
    }

    grid = GridSearchCV(
        estimator=RandomForestClassifier(random_state=42, n_jobs=-1),
        param_grid=param_grid,
        cv=5,
        scoring="accuracy",
        n_jobs=-1
    )

    grid.fit(X_train, y_train)

    print("Melhores parâmetros:", grid.best_params_)
    print("Melhor score RF CV:", grid.best_score_)

    return grid.best_estimator_


def treinar_xgboost_otimizado(X_train, y_train):
    """
    XGBoost otimizado com search mais completo e early stopping.
    """
    print("\n=== Ajustando XGBoost (RandomizedSearchCV + Early Stopping) ===")

    param_dist = {
        "n_estimators": [50, 100, 200],
        "learning_rate": [0.01, 0.03, 0.05, 0.1],
        "max_depth": [2, 3, 4, 5],
        "min_child_weight": [3, 5, 7],
        "gamma": [0, 0.1, 0.5, 1],
        "subsample": [0.6, 0.7, 0.8, 1.0],
        "colsample_bytree": [0.6, 0.7, 0.8, 1.0],
        "reg_alpha": [0, 0.001, 0.01, 0.1],
        "reg_lambda": [1, 1.5, 2, 3],
    }

    modelo_base = XGBClassifier(
        objective="multi:softprob",
        eval_metric="mlogloss",
        random_state=42,
        n_jobs=-1
    )

    random_search = RandomizedSearchCV(
        estimator=modelo_base,
        param_distributions=param_dist,
        n_iter=40,                 
        scoring="accuracy",
        cv=5,
        n_jobs=-1,
        random_state=42,
        verbose=1
    )

    random_search.fit(X_train, y_train)

    print("Melhores parâmetros:", random_search.best_params_)
    print("Melhor score CV:", random_search.best_score_)

    return random_search.best_estimator_


# ===================================================================
# Geração do arquivo de submissão
# ===================================================================

def gerar_submissao(model, X_test, test_ids, label_encoder, base_path="dataset",
                    nome_arquivo="submission.csv"):
    """
    Gera submissão no formato: 0=STRESS, 1=AEROBIC, 2=ANAEROBIC.
    """
    print("\n--- Gerando arquivo de submissão ---")

    # obter probabilidades brutas do modelo
    if hasattr(model, "predict_proba"):
        y_pred_proba = model.predict_proba(X_test)
    else:
        preds = model.predict(X_test)
        n_classes_model = len(label_encoder.classes_)
        y_pred_proba = np.zeros((len(preds), n_classes_model))
        for i, p in enumerate(preds):
            # fallback manual
            y_pred_proba[i, p] = 1.0

    # recupera index
    classes_modelo = list(label_encoder.classes_)
    
    idx_stress = classes_modelo.index("STRESS")
    idx_aerobic = classes_modelo.index("AEROBIC")
    idx_anaerobic = classes_modelo.index("ANAEROBIC")
    
    # montagem dataframe
    submission = pd.DataFrame()
    submission["Id"] = test_ids
    
    # atribui index
    submission["Predicted_0"] = y_pred_proba[:, idx_stress] # STRESS
    submission["Predicted_1"] = y_pred_proba[:, idx_aerobic] # AEROBIC
    submission["Predicted_2"] = y_pred_proba[:, idx_anaerobic] # ANAEROBIC

    cols_pred = ["Predicted_0", "Predicted_1", "Predicted_2"]
    submission[cols_pred] = submission[cols_pred].div(submission[cols_pred].sum(axis=1), axis=0)

    # salvar
    output_path = os.path.join(base_path, nome_arquivo)
    submission.to_csv(output_path, index=False, float_format="%.6f")

    print(f"\nSubmissão salva em: {output_path}")
    print(submission.head())

    return submission
