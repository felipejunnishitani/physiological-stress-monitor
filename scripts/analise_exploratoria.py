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
#
# Arquivo com todas as funções e códigos referentes à análise exploratória

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scripts.preprocessamento import gerar_dataset


# ================================================================
# Carregar dataset + resumo geral
# ================================================================
def carregar_e_resumir(base_path: str) -> pd.DataFrame:
    """
    Carrega o dataset consolidado gerado pelo preprocessamento e exibe:
        - head()
        - info()
        - describe()
    """
    print("\n--- Carregando dataset ---")
    df = gerar_dataset(base_path)

    print("\nPrimeiras linhas do dataset:")
    display(df.head())

    print("\nInformações gerais:")
    df_info = df.info()          # info imprime sozinho
    print(df_info)

    print("\nEstatísticas descritivas:")
    display(df.describe(include="all"))

    return df


# ================================================================
# Medidas descritivas
# ================================================================
def medidas_descritivas(df: pd.DataFrame):
    """Mostra distribuição das classes e valores ausentes."""
    print("\n--- Medidas Descritivas ---")

    if "classe" in df.columns:
        print("\nDistribuição das classes:")
        display(df["classe"].value_counts())

        plt.figure(figsize=(6, 4))
        ax = sns.countplot(x="classe", data=df, hue="classe", palette="Set2", legend=False)
        plt.title("Distribuição das Classes")
        plt.show()
    else:
        print("Coluna 'classe' não encontrada no dataset.")

    print("\nValores ausentes por coluna:")
    display(df.isnull().sum())


# ================================================================
# Boxplots e Histogramas
# ================================================================
def boxplots_e_histogramas(df: pd.DataFrame, top_n=6):
    """
    Gera boxplots e histogramas apenas para os atributos numéricos mais informativos, com gráficos organizados em grade.
    
    Critérios de seleção:
    - Top N atributos com maior variância
    - Top N atributos mais relevantes para a classe (ANOVA F-score)
    - Top N atributos com maior correlação absoluta entre si
    
    Isso evita gerar dezenas de gráficos irrelevantes e torna a EDA mais objetiva.
    """
    print("\n--- Boxplots e Histogramas ---")

    # ====================================
    # 1. Selecionar atributos numéricos
    # ====================================
    num_df = df.select_dtypes(include=np.number)

    if num_df.empty:
        print("Nenhuma coluna numérica encontrada.")
        return

    # ============================
    # 2. Ranking por variância
    # ============================
    variancias = num_df.var().sort_values(ascending=False)
    top_var = variancias.head(top_n).index.tolist()

    # ================================================
    # 3. Ranking por relevância com a classe (ANOVA)
    # ================================================
    top_fscore = []
    if "classe" in df.columns:
        try:
            from sklearn.feature_selection import f_classif
            f_vals, _ = f_classif(num_df, df["classe"])
            f_scores = pd.Series(f_vals, index=num_df.columns)
            top_fscore = f_scores.sort_values(ascending=False).head(top_n).index.tolist()
        except:
            pass

    # ============================
    # 4. Ranking por correlação
    # ============================
    corr_matrix = num_df.corr()
    corr_pairs = corr_matrix.abs().unstack()
    corr_pairs = corr_pairs[corr_pairs < 1]           # tira diagonais
    corr_pairs = corr_pairs.sort_values(ascending=False)
    top_corr = list(dict.fromkeys([a for a, b in corr_pairs.index[:top_n]]))

    # =================================================
    # 5. Seleção final dos atributos mais importantes
    # =================================================
    atributos_escolhidos = list(dict.fromkeys(top_var + top_fscore + top_corr))
    atributos_escolhidos = atributos_escolhidos[:top_n]  # limitar ao desejado

    print(f"\nAtributos selecionados para visualização: {atributos_escolhidos}")

    # -------------------------------------------------------------------------
    # BOX-PLOTS (em grid 2 × N/2)
    # -------------------------------------------------------------------------
    n_cols = 2
    n_rows = int(np.ceil(len(atributos_escolhidos) / n_cols))

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(14, 4 * n_rows))
    axes = axes.flatten()

    for ax, col in zip(axes, atributos_escolhidos):
        sns.boxplot(x=df[col], ax=ax, color="skyblue")
        ax.set_title(f"Boxplot – {col}")

    # Apagar subplots vazios
    for i in range(len(atributos_escolhidos), len(axes)):
        fig.delaxes(axes[i])

    plt.tight_layout()
    plt.show()

    # -------------------------------------------------------------------------
    # HISTOGRAMAS (em grid)
    # -------------------------------------------------------------------------
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(14, 4 * n_rows))
    axes = axes.flatten()

    for ax, col in zip(axes, atributos_escolhidos):

        if "classe" in df.columns:
            # Histograma por classe
            for classe in df["classe"].unique():
                subset = df[df["classe"] == classe][col]
                ax.hist(subset, bins=25, alpha=0.5, label=str(classe))
            ax.legend()
        else:
            ax.hist(df[col], bins=25, alpha=0.7)

        ax.set_title(f"Histograma – {col}")
        ax.set_xlabel(col)
        ax.set_ylabel("Frequência")

    # Apagar subplots vazios
    for i in range(len(atributos_escolhidos), len(axes)):
        fig.delaxes(axes[i])

    plt.tight_layout()
    plt.show()



# ================================================================
# Correlação entre atributos
# ================================================================
def correlacao_atributos(df: pd.DataFrame):
    """Plota a matriz de correlação para todos os atributos numéricos."""
    print("\n--- Matriz de Correlação ---")

    num_df = df.select_dtypes(include=np.number)

    if num_df.empty:
        print("Nenhuma coluna numérica encontrada.")
        return

    corr = num_df.corr()

    plt.figure(figsize=(12, 8))
    sns.heatmap(corr, cmap="coolwarm", annot=False)
    plt.title("Matriz de Correlação entre Atributos Numéricos")
    plt.show()

    # Correlações ordenadas
    corr_pairs = corr.unstack().drop_duplicates()
    corr_pairs = corr_pairs[corr_pairs.abs() < 1]  # remove diagonais
    corr_pairs = corr_pairs.reindex(corr_pairs.abs().sort_values(ascending=False).index)

    print("\n Maiores correlações:")
    display(corr_pairs.head(10))


# ================================================================
# Pairplot com amostra
# ================================================================
def pairplot_amostrado(df: pd.DataFrame, n_amostras=300):
    """
    Gera pairplot com amostra reduzida, selecionando automaticamente os atributos mais relevantes para motivar as features de interação.
    """

    print("\n--- Pairplot Amostrado ---")

    atributos_relevantes = []

    # HR / ACC -> índice de esforço cardíaco relativo ao movimento
    if "hr_mean" in df.columns and "acc_energy" in df.columns:
        atributos_relevantes += ["hr_mean", "acc_energy"]

    # ACC x Temp Slope -> movimento vs resposta térmica periférica
    if "acc_energy" in df.columns and "temp_slope" in df.columns:
        atributos_relevantes += ["acc_energy", "temp_slope"]

    # EDA x Temp Mean -> ativação autonômica (suor + temperatura)
    if "eda_mean" in df.columns and "temp_mean" in df.columns:
        atributos_relevantes += ["eda_mean", "temp_mean"]

    # HRV × EDA std -> reatividade simpática
    if "hrv_rmssd" in df.columns and "eda_std" in df.columns:
        atributos_relevantes += ["hrv_rmssd", "eda_std"]

    # Remove duplicatas mantendo a ordem
    atributos_relevantes = list(dict.fromkeys(atributos_relevantes))

    if len(atributos_relevantes) == 0:
        print("Nenhum dos atributos necessários para interações foi encontrado.")
        return

    # Adiciona a classe se existir
    if "classe" in df.columns:
        cols = atributos_relevantes + ["classe"]
    else:
        cols = atributos_relevantes

    # Amostragem segura
    sample = df.sample(n=min(n_amostras, len(df)), random_state=42)
    print(f"--- Gerando pairplot com {len(sample)} amostras e {len(atributos_relevantes)} atributos ---")

    sns.pairplot(
        sample[cols],
        hue="classe" if "classe" in df.columns else None,
        height=2.0,
        diag_kind="kde"
    )
    plt.show()



# ==============================================================
# Análise de séries temporais
# ==============================================================
def plot_series_temporais(df, colunas):
    """
    Plota as séries temporais em um layout de subplots.
    """
    print("\n--- Séries Temporais (Subplots) ---")
    
    # Filtra colunas que realmente existem
    cols_validas = [col for col in colunas if col in df.columns]
    if not cols_validas:
        print("Nenhuma das colunas especificadas foi encontrada.")
        return

    # Define o layout: número de colunas fixo (2) e calcula as linhas
    ncols = 2
    nrows = int(np.ceil(len(cols_validas) / ncols))
    
    # Define o tamanho total da figura
    fig, axes = plt.subplots(nrows, ncols, figsize=(15, 4 * nrows))
    
    # Transforma 'axes' em um array 1D para facilitar a iteração
    axes = axes.flatten() if nrows > 1 else [axes]

    for i, col in enumerate(cols_validas):
        # Plota no subplot atual
        axes[i].plot(df[col].values)
        axes[i].set_title(f"{col}", fontsize=12)
        axes[i].set_xlabel("Tempo (amostra)")
        axes[i].ticklabel_format(style='sci', axis='y', scilimits=(0, 0)) # Formata para notação científica, se necessário
    
    # Remove subplots vazios, se houver
    for j in range(len(cols_validas), len(axes)):
        fig.delaxes(axes[j])
        
    plt.suptitle("Séries Temporais dos Atributos Fisiológicos", fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.96]) # Ajusta o layout para dar espaço ao suptitle
    plt.show()


# ==============================================================
# Scatter plot duplo
# ==============================================================
def scatter_duplo(df: pd.DataFrame, lista_pares: list):
    """
    Gera scatter plots de pares de atributos em um layout de subplots.
    """
    print("\n--- Scatter Plots Duplos (Subplots) ---")
    
    n_graficos = len(lista_pares)
    ncols = 3  # 3 colunas por linha
    nrows = int(np.ceil(n_graficos / ncols))
    
    fig, axes = plt.subplots(nrows, ncols, figsize=(18, 5 * nrows))
    
    # Transforma 'axes' em um array 1D para fácil iteração
    axes = axes.flatten() if nrows > 1 else [axes]
    
    # Verifica a existência da coluna 'classe' para o hue
    use_hue = "classe" if "classe" in df.columns else None

    for i, (x, y) in enumerate(lista_pares):
        if x not in df.columns or y not in df.columns:
            print(f"Pares {x} ou {y} não encontrados.")
            continue
            
        sns.scatterplot(
            data=df, 
            x=x, 
            y=y, 
            hue=use_hue, 
            ax=axes[i], 
            alpha=0.6,
            palette="viridis" if use_hue else None
        )
        axes[i].set_title(f"{x} vs {y}", fontsize=12)
        axes[i].legend(title="Classe")
    
    # Remove subplots vazios, se houver
    for j in range(n_graficos, len(axes)):
        fig.delaxes(axes[j])

    plt.suptitle("Scatter Plots de Pares Fisiológicos", fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()