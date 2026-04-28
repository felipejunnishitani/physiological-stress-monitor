# ################################################################
# PROJETO FINAL
#
# Universidade Federal de Sao Carlos (UFSCAR)
# Departamento de Computacao - Sorocaba (DComp-So)
# Disciplina: Aprendizado de Maquina
# Prof. Tiago A. Almeida
#
# Nome: Anne Mari Suenaga Sakai
# RA: 822304
# 
# Nome: Felipe Jun Nishitani
# RA: 822353
# ################################################################

# Arquivo com todas as funcoes e codigos referentes ao preprocessamento


import os
import math
import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt
from scipy.stats import skew, kurtosis
from sklearn.preprocessing import StandardScaler
from scipy.fft import fft
from collections import Counter


# ===================================================================
# Utilitários
# ===================================================================
def _safe_read_csv(path, header=None):
    """Lê csv sem assumir cabeçalho; retorna DataFrame com todas as colunas brutas."""
    try:
        return pd.read_csv(path, header=header)
    except Exception:
        # fallback mais permissivo (p.e. linhas com timestamps repetidos)
        return pd.read_csv(path, header=None, engine="python", error_bad_lines=False)


def _to_numeric_array(series):
    """Tenta converter uma Series pandas para float, removendo valores não-convertíveis."""
    return pd.to_numeric(series, errors="coerce").dropna().astype(float).values


def _shannon_entropy(arr, bins=30):
    """Estimativa simples de entropia de Shannon via histograma."""
    if len(arr) == 0:
        return np.nan
    hist, _ = np.histogram(arr, bins=bins, density=True)
    # eliminar zeros
    hist = hist[hist > 0]
    return -np.sum(hist * np.log2(hist))


# ===================================================================
# HRV (IBI)
# ===================================================================
def extrair_hrv(df_ibi):
    """
    Recebe df_ibi (DataFrame do IBI.csv).
    Assume que a coluna 1 (index 1) contém IBI em segundos (caso do seu dataset).
    Retorna: (rmssd, sdnn)
    """
    try:
        ibi = _to_numeric_array(df_ibi.iloc[:, 1])
    except Exception:
        ibi = np.array([])

    if len(ibi) < 3:
        return np.nan, np.nan

    sdnn = float(np.std(ibi, ddof=0))
    diffs = np.diff(ibi)
    rmssd = float(np.sqrt(np.mean(diffs**2))) if len(diffs) > 0 else np.nan
    return rmssd, sdnn


# ===================================================================
# EDA (tonic / phasic simplificado)
# ===================================================================
def extrair_eda_components(eda_series):
    """
    Extrai componentes tonic (passa-baixa) e phasic (resáduo).
    Entrada: pd.Series ou array-like com EDA (em µS).
    Retorna: (tonic_mean, phasic_mean, eda_mean, eda_std)
    Observação: filtro simples Butterworth de baixa frequência para tonic.
    """
    arr = _to_numeric_array(pd.Series(eda_series))
    if len(arr) < 10:
        return np.nan, np.nan, np.nan, np.nan

    # Butterworth passa-baixa (tonic). Normalizar frequência por Nyquist assume taxa unknown;
    # usamos corte baixo absoluto (0.05) apropriado para sinais lentos — funciona em amostragens típicas.
    try:
        b, a = butter(2, 0.05, btype="low", analog=False)
        tonic = filtfilt(b, a, arr)
    except Exception:
        # caso o filtro falhe (p.ex. poucos pontos), usar média móvel simples
        window = max(3, int(len(arr) * 0.05))
        tonic = pd.Series(arr).rolling(window=window, min_periods=1, center=True).mean().values

    phasic = arr - tonic
    return float(np.mean(tonic)), float(np.mean(phasic)), float(np.mean(arr)), float(np.std(arr))



# ===================================================================
# ACC (x,y,z) -> magnitude, energia, entropia
# ===================================================================
def extrair_acc_features(df_acc):
    """
    Recebe df_acc (DataFrame). Espera três colunas numéricas (x,y,z) possivelmente sem cabeçalho.
    Retorna: (mag_mean, mag_std, energy, entropy)
    """
    # tentar forçar 3 colunas numéricas
    arr = df_acc.copy()
    # converter tudo para número, descartar colunas não numéricas
    for c in arr.columns:
        arr[c] = pd.to_numeric(arr[c], errors="coerce")
    num_cols = arr.select_dtypes(include=[np.number]).columns.tolist()
    if len(num_cols) < 3:
        # se menos de 3 colunas numéricas, tentar usar primeiras 3 colunas
        if arr.shape[1] >= 3:
            arr = arr.iloc[:, :3].apply(pd.to_numeric, errors="coerce")
            num_cols = arr.select_dtypes(include=[np.number]).columns.tolist()
        else:
            return np.nan, np.nan, np.nan, np.nan

    data = arr[num_cols[:3]].dropna().values.astype(float)
    if data.size == 0:
        return np.nan, np.nan, np.nan, np.nan

    x, y, z = data[:, 0], data[:, 1], data[:, 2]
    magnitude = np.sqrt(x**2 + y**2 + z**2)

    mag_mean = float(np.mean(magnitude))
    mag_std = float(np.std(magnitude))
    energy = float(np.sum(magnitude**2) / len(magnitude))
    entropy = float(_shannon_entropy(magnitude, bins=30))

    return mag_mean, mag_std, energy, entropy


# ===================================================================
# TEMP features
# ===================================================================
def extrair_temp_features(temp_series):
    """
    Retorna temp_mean, temp_std, temp_slope
    """
    arr = _to_numeric_array(pd.Series(temp_series))
    
    # filtra ruído físico
    arr_limpo = arr[(arr > 10) & (arr < 60)]
    
    # retorna nan se todos os dados eram ruído
    if len(arr_limpo) < 3:
        return np.nan, np.nan, np.nan

    mean_temp = float(np.mean(arr_limpo))
    std_temp = float(np.std(arr_limpo))
    
    # calculo slope
    try:
        slope = float(np.polyfit(np.arange(len(arr_limpo)), arr_limpo, 1)[0])
    except:
        slope = 0.0
        
    return mean_temp, std_temp, slope


# ===================================================================
# HR features (quando HR.csv estiver presente)
# ===================================================================
def extrair_hr_features(hr_series):
    """
    Retorna hr_mean, hr_std
    """
    arr = _to_numeric_array(pd.Series(hr_series))
    
    # filtra ruído físico
    arr_limpo = arr[(arr >= 30) & (arr <= 220)]
    
    if len(arr_limpo) == 0:
        return np.nan, np.nan
        
    return float(np.mean(arr_limpo)), float(np.std(arr_limpo))


# ===================================================================
# # Função auxiliar: extrai estatísticas simples (compatibilidade)
# ===================================================================
def estatisticas_basicas_numeric(df):
    """
    Recebe dataframe com colunas numéricas e retorna mean/std para cada coluna
    (usa para compatibilidade/diagnóstico).
    Retorna dict com chaves "<col>_mean", "<col>_std".
    """
    res = {}
    num = df.select_dtypes(include=[np.number])
    for c in num.columns:
        res[f"{c}_mean"] = float(num[c].mean())
        res[f"{c}_std"] = float(num[c].std())
    return res


# ===================================================================-
# Geração consolidada do dataset (treino/teste)
# ===================================================================-
def gerar_dataset(base_path: str, mode="train"):
    """
    Gera dataset consolidado (treino ou teste) a partir da pasta base.
    Espera:
      base_path/wearables/U_xxxxx/{ACC.csv, TEMP.csv, IBI.csv, HR.csv, EDA.csv, BVP.csv, tags.csv}
    mode: "train" ou "test" (usa train.csv/test.csv para obter Ids/labels se existir)
    Retorna DataFrame com colunas com as features definidas no cabeçalho desta versão.
    """
    wearables_path = os.path.join(base_path, "wearables")
    ids = []
    labels = {}

    # se houver train.csv, use-o para labels; caso contrário, processa todos os U_*
    csv_ref = os.path.join(base_path, f"{mode}.csv")
    if os.path.exists(csv_ref):
        df_ref = pd.read_csv(csv_ref)
        for _, r in df_ref.iterrows():
            ids.append(str(r["Id"]))
            if "Label" in r:
                labels[str(r["Id"])] = r["Label"]
    else:
        # coletar todos os folders U_*
        ids = [name for name in os.listdir(wearables_path) if os.path.isdir(os.path.join(wearables_path, name))]

    data_rows = []
    for user_id in ids:
        user_folder = os.path.join(wearables_path, user_id)
        if not os.path.isdir(user_folder):
            continue

        user_data = {"Id": user_id}
        if user_id in labels:
            user_data["classe"] = labels[user_id]

        # listar arquivos esperados
        files = {os.path.splitext(f)[0].upper(): os.path.join(user_folder, f) for f in os.listdir(user_folder)}

        # 1) ACC
        if "ACC" in files:
            try:
                df_acc = _safe_read_csv(files["ACC"], header=None)
                mag_mean, mag_std, energy, entropy = extrair_acc_features(df_acc)
                user_data["acc_mag_mean"] = mag_mean
                user_data["acc_mag_std"] = mag_std
                user_data["acc_energy"] = energy
                user_data["acc_entropy"] = entropy
            except Exception as e:
                # não falhar todo processo
                user_data["acc_mag_mean"] = np.nan
                user_data["acc_mag_std"] = np.nan
                user_data["acc_energy"] = np.nan
                user_data["acc_entropy"] = np.nan

        # 2) IBI -> HRV
        if "IBI" in files:
            try:
                df_ibi = _safe_read_csv(files["IBI"], header=None)
                rmssd, sdnn = extrair_hrv(df_ibi)
                user_data["hrv_rmssd"] = rmssd
                user_data["hrv_sdnn"] = sdnn
            except Exception:
                user_data["hrv_rmssd"] = np.nan
                user_data["hrv_sdnn"] = np.nan

        # 3) EDA
        if "EDA" in files:
            try:
                df_eda = _safe_read_csv(files["EDA"], header=None)
                # extração usando a primeira coluna detectada
                tonic_mean, phasic_mean, eda_mean, eda_std = extrair_eda_components(df_eda.iloc[:, 0])
                user_data["eda_tonic_mean"] = tonic_mean
                user_data["eda_phasic_mean"] = phasic_mean
                user_data["eda_mean"] = eda_mean
                user_data["eda_std"] = eda_std
            except Exception:
                user_data["eda_tonic_mean"] = np.nan
                user_data["eda_phasic_mean"] = np.nan
                user_data["eda_mean"] = np.nan
                user_data["eda_std"] = np.nan

        # 4) TEMP
        if "TEMP" in files:
            try:
                df_temp = _safe_read_csv(files["TEMP"], header=None)
                t_mean, t_std, t_slope = extrair_temp_features(df_temp.iloc[:, 0])
                user_data["temp_mean"] = t_mean
                user_data["temp_std"] = t_std
                user_data["temp_slope"] = t_slope
            except Exception:
                user_data["temp_mean"] = np.nan
                user_data["temp_std"] = np.nan
                user_data["temp_slope"] = np.nan

        # 5) HR
        if "HR" in files:
            try:
                df_hr = _safe_read_csv(files["HR"], header=None)
                hr_mean, hr_std = extrair_hr_features(df_hr.iloc[:, 0])
                user_data["hr_mean"] = hr_mean
                user_data["hr_std"] = hr_std
            except Exception:
                user_data["hr_mean"] = np.nan
                user_data["hr_std"] = np.nan

        # 6) BVP (opcional: média como fallback)
        if "BVP" in files:
            try:
                df_bvp = _safe_read_csv(files["BVP"], header=None)
                bvp = _to_numeric_array(df_bvp.iloc[:, 0]) if df_bvp.shape[1] >= 1 else np.array([])
                user_data["bvp_mean"] = float(np.mean(bvp)) if bvp.size > 0 else np.nan
            except Exception:
                user_data["bvp_mean"] = np.nan


        data_rows.append(user_data)

    df_features = pd.DataFrame(data_rows)
    # organizar colunas (Id, classe se existir, e features ordenadas)
    cols = ["Id"]
    if "classe" in df_features.columns:
        cols.append("classe")
    # adicionar o resto em ordem alfabética (previsível)
    other = sorted([c for c in df_features.columns if c not in cols])
    cols.extend(other)
    df_features = df_features[cols]

    return df_features


# ===================================================================
# Tratamento de valores ausentes (simples e robusto)
# ===================================================================
def tratar_valores_ausentes(df: pd.DataFrame, strategy="median"):
    """
    strategy: 'median' (numéricas) e moda (categóricas)
    Retorna DF com valores preenchidos.
    """
    out = df.copy()
    
    out = out.replace("?", np.nan)
    
    for col in out.columns:
        try:
            out[col] = pd.to_numeric(out[col])
        except (ValueError, TypeError):
            pass
        # dado numérico
        if pd.api.types.is_numeric_dtype(out[col]):
            if strategy == "median":
                # calcula mediana
                valor = out[col].median()
            else:
                # se nao calcula média
                valor = out[col].mean()
            
            out[col] = out[col].fillna(valor)
            
        # dado categórico
        else:
            # calcula moda
            moda = out[col].mode()
            valor = moda[0] if not moda.empty else "Desconhecido"
            
            out[col] = out[col].fillna(valor)
    return out


# ===================================================================
# Remoção de atributos redundantes
# ===================================================================
def remover_atributos_redundantes(df: pd.DataFrame):
    """
    Remove colunas que são linearmente dependentes ou redundantes baseadas na análise de correlação (Heatmap/Pairplot).
    """

    # lista de colunas para remover
    cols_to_drop = ['eda_tonic_mean','acc_mag_mean']
    
    # remove as colunas
    df_out = df.drop(columns=cols_to_drop, errors='ignore')
    
    print(f"--- Atributos removidos: {cols_to_drop} ---")
    return df_out



# ===================================================================
# Transformação logarítmica
# ===================================================================
def aplicar_transformacao_log(df: pd.DataFrame):
    """
    Aplica transformação Logarítmica (np.log1p) em atributos com distribuição altamente assimétrica.
    
    Isso 'normaliza' a distribuição, permitindo que o Z-Score funcione corretamente para detecção de outliers e melhora o treino de modelos.
    """
    df_out = df.copy()

    # lista de colunas para aplicar a transformação log
    cols_to_log = [
        'acc_energy', 
        'acc_mag_std', 
        'acc_entropy', 
        'eda_mean', 
        'eda_phasic_mean'
    ]

    print(f"--- Aplicando Log1p em: {cols_to_log} ---")

    for col in cols_to_log:
        # aplica log(x + 1)
        df_out[col] = np.log1p(df_out[col])
            
    return df_out


# ===================================================================
# Limitar de outliers (z-score)
# ===================================================================
def limitar_outliers_zscore(df: pd.DataFrame, zmax=3.0):
    """
    Limita valores extremos (capping) usando Z-score por coluna, sem remover linhas.
    """

    df_new = df.copy()
    num = df_new.select_dtypes(include=[np.number])

    for col in num.columns:
        # calculo do z-score
        col_z = (df_new[col] - df_new[col].mean()) / df_new[col].std()

        # aplica limite
        df_new[col] = np.where(col_z > zmax, df_new[col].mean() + zmax * df_new[col].std(), np.where(col_z < -zmax, df_new[col].mean() - zmax * df_new[col].std(), df_new[col]))
    return df_new


# ===================================================================
# Criação de features de interação
# ===================================================================
def criar_features_interacao(df: pd.DataFrame):
    """
    Cria novas features combinando sensores diferentes para ajudar
    modelos lineares e árvores a distinguir 'Stress' de 'Exercício'.
    """
    df_out = df.copy()
    cols = df_out.columns

    # índice de stress físico (HR / ACC)
    if "hr_mean" in cols and "acc_energy" in cols:
        df_out["inter_hr_acc_ratio"] = df_out["hr_mean"] / (df_out["acc_energy"] + 1.0)

    # movimento × resposta térmica (ACC × Temp Slope)
    if "acc_energy" in cols and "temp_slope" in cols:
        df_out["inter_acc_temp"] = df_out["acc_energy"] * df_out["temp_slope"]

    # ativação autonômica (EDA × Temperatura)
    if "eda_mean" in cols and "temp_mean" in cols:
        df_out["inter_eda_temp_mult"] = df_out["eda_mean"] * df_out["temp_mean"]

    # reatividade fisiológica combinada (HRV × EDA std)
    if "hrv_rmssd" in cols and "eda_std" in cols:
        df_out["inter_hrv_eda"] = df_out["hrv_rmssd"] * df_out["eda_std"]


    print("--- Features de interação criadas ---")
    return df_out



# ===================================================================
# Preparar dados (encode + normalização + alinhamento treino/teste)
# ===================================================================
def preparar_dados(df_train: pd.DataFrame, df_test: pd.DataFrame):
    """
    Prepara dados garantindo alinhamento entre treino e teste,
    mesmo após criação/remoção de features.
    """

    from sklearn.preprocessing import LabelEncoder, StandardScaler

    df_train = df_train.copy()
    df_test = df_test.copy()

    # label encode da classe
    if "classe" in df_train.columns:
        le = LabelEncoder()
        y_train = le.fit_transform(df_train["classe"].astype(str))
    else:
        le = None
        y_train = None

    # remove id e classe
    X_train = df_train.drop(columns=["Id", "classe"], errors="ignore").copy()
    X_test = df_test.drop(columns=["Id", "classe"], errors="ignore").copy()
    test_ids = df_test["Id"].copy() if "Id" in df_test.columns else None

    # trata valor ausente
    X_train = tratar_valores_ausentes(X_train)
    X_test = tratar_valores_ausentes(X_test)

    # alinha colunas entre treino e teste
    colunas_treino = X_train.columns

    # adiciona no teste colunas que só existem no treino
    for col in colunas_treino:
        if col not in X_test.columns:
            X_test[col] = 0

    # remove do teste colunas que não existem no treino
    for col in X_test.columns:
        if col not in colunas_treino:
            X_test = X_test.drop(columns=[col])

    # reordena teste
    X_test = X_test[colunas_treino]

    # normaliza  com StandardScaler
    scaler = StandardScaler()

    X_train_vals = scaler.fit_transform(X_train)
    X_test_vals = scaler.transform(X_test)

    # reconstroi df padronizados
    X_train_scaled = pd.DataFrame(X_train_vals, columns=colunas_treino, index=X_train.index)
    X_test_scaled = pd.DataFrame(X_test_vals,  columns=colunas_treino, index=X_test.index)

    return X_train_scaled, y_train, X_test_scaled, test_ids, le, scaler
