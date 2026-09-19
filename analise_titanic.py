# -*- coding: utf-8 -*-
"""
PROJETO DE ANÁLISE ESTATÍSTICA — BASE DE DADOS TITANIC
Probabilidade e Estatística

Este script realiza a análise estatística descritiva da base de dados do
Titanic (891 passageiros): tratamento de dados ausentes, análise de
variáveis qualitativas e quantitativas, comparação entre grupos e uma
investigação livre sobre o título de tratamento presente no nome dos
passageiros.

Pode ser executado célula por célula (no VS Code/Jupyter os blocos
"# %%" viram células) ou por completo com: python analise_titanic.py

Todos os gráficos são salvos automaticamente na pasta "graficos/" e o
detalhamento de cada resultado, com a interpretação, está no relatório
(Relatorio_Titanic.docx).
"""

# %% 1. IMPORTS E CONFIGURAÇÃO
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import os

sns.set_theme(style="whitegrid")
plt.rcParams["figure.dpi"] = 110
os.makedirs("graficos", exist_ok=True)


def salvar(fig, nome):
    """Salva a figura na pasta graficos/ com nome padronizado."""
    fig.savefig(f"graficos/{nome}.png", bbox_inches="tight")
    plt.close(fig)


# %% 2. CARREGAR OS DADOS
df = pd.read_csv("titanic.csv")

print("Dimensões da base:", df.shape)
print("\nPrimeiras linhas:")
print(df.head())


# ======================================================================
# ETAPA 1 — CONHECENDO A BASE DE DADOS
# ======================================================================

# %% 3. QUANTAS OBSERVAÇÕES E VARIÁVEIS?
n_obs, n_var = df.shape
print(f"\nA base tem {n_obs} observações (passageiros) e {n_var} variáveis.")
print("Cada LINHA representa um passageiro do Titanic.")
print("Cada COLUNA representa uma característica desse passageiro.")

# %% 4. CLASSIFICAÇÃO DAS VARIÁVEIS
# Preencham/ajustem esta classificação no relatório — ela já está correta
# para esta base, mas expliquem o PORQUÊ de cada tipo no texto.
classificacao = {
    "PassengerId": "Qualitativa nominal (identificador)",
    "Survived": "Qualitativa nominal (0 = não sobreviveu, 1 = sobreviveu)",
    "Pclass": "Qualitativa ordinal (1ª, 2ª, 3ª classe — há uma ordem)",
    "Name": "Qualitativa nominal (texto)",
    "Sex": "Qualitativa nominal",
    "Age": "Quantitativa contínua",
    "SibSp": "Quantitativa discreta (nº de irmãos/cônjuges a bordo)",
    "Parch": "Quantitativa discreta (nº de pais/filhos a bordo)",
    "Ticket": "Qualitativa nominal",
    "Fare": "Quantitativa contínua",
    "Cabin": "Qualitativa nominal (muitos valores ausentes)",
    "Embarked": "Qualitativa nominal (porto de embarque)",
}
tabela_classificacao = pd.DataFrame.from_dict(
    classificacao, orient="index", columns=["Classificação"]
)
print("\nClassificação das variáveis:")
print(tabela_classificacao)

# %% 5. DADOS AUSENTES
ausentes = df.isnull().sum()
ausentes_pct = (df.isnull().mean() * 100).round(2)
tabela_ausentes = pd.DataFrame({"Ausentes": ausentes, "% do total": ausentes_pct})
tabela_ausentes = tabela_ausentes[tabela_ausentes["Ausentes"] > 0]
print("\nVariáveis com dados ausentes:")
print(tabela_ausentes)

# %% 6. TRATAMENTO DOS DADOS AUSENTES (decisão sugerida — justifiquem a sua no relatório)
df_tratado = df.copy()

# Age (~20% ausente): preenchido com a MEDIANA por ser mais robusta a
# valores extremos do que a média.
df_tratado["Age"] = df_tratado["Age"].fillna(df_tratado["Age"].median())

# Cabin (~77% ausente): quase toda a coluna está vazia, então em vez de
# descartar a informação, criamos uma variável binária indicando se a
# cabine foi registrada — isso por si só pode ser interessante (pode
# estar relacionado à classe do passageiro).
df_tratado["Cabine_Informada"] = df_tratado["Cabin"].notnull().map({True: "Sim", False: "Não"})

# Embarked (2 ausentes): preenchido com a MODA (valor mais frequente),
# já que são só 2 casos e a moda é a escolha padrão para poucas ausências.
df_tratado["Embarked"] = df_tratado["Embarked"].fillna(df_tratado["Embarked"].mode()[0])

print("\nDados ausentes após tratamento:")
print(df_tratado[["Age", "Embarked", "Cabine_Informada"]].isnull().sum())

df_tratado.to_csv("titanic_tratado.csv", index=False)
print("\nBase tratada salva em titanic_tratado.csv")


# ======================================================================
# ETAPA 2 — VARIÁVEIS QUALITATIVAS
# ======================================================================

def tabela_frequencia(coluna, rotulo=None):
    """Gera tabela de frequência absoluta, relativa e percentual."""
    freq_abs = coluna.value_counts()
    freq_rel = coluna.value_counts(normalize=True).round(4)
    freq_pct = (freq_rel * 100).round(2)
    tabela = pd.DataFrame({
        "Freq. absoluta": freq_abs,
        "Freq. relativa": freq_rel,
        "Percentual (%)": freq_pct,
    })
    print(f"\n--- {rotulo or coluna.name} ---")
    print(tabela)
    return tabela


def grafico_barras(coluna, titulo, nome_arquivo, rotulos=None):
    fig, ax = plt.subplots(figsize=(6, 4))
    contagem = coluna.value_counts()
    if rotulos:
        contagem.index = [rotulos.get(i, i) for i in contagem.index]
    sns.barplot(x=contagem.index, y=contagem.values, hue=contagem.index,
                ax=ax, palette="viridis", legend=False)
    ax.set_title(titulo)
    ax.set_ylabel("Número de passageiros")
    ax.set_xlabel("")
    for i, v in enumerate(contagem.values):
        ax.text(i, v + 5, str(v), ha="center")
    salvar(fig, nome_arquivo)


# %% 7. SEXO
tabela_frequencia(df_tratado["Sex"], "Sexo")
grafico_barras(df_tratado["Sex"], "Distribuição por sexo", "sexo_barras",
               rotulos={"male": "Homem", "female": "Mulher"})
# A interpretação desse resultado está no relatório (seção 3.1).

# %% 8. CLASSE (Pclass)
tabela_frequencia(df_tratado["Pclass"], "Classe")
grafico_barras(df_tratado["Pclass"], "Distribuição por classe", "classe_barras",
               rotulos={1: "1ª classe", 2: "2ª classe", 3: "3ª classe"})
# A interpretação desse resultado está no relatório (seção 3.2).

# %% 9. SOBREVIVÊNCIA
tabela_frequencia(df_tratado["Survived"], "Sobrevivência")
grafico_barras(df_tratado["Survived"], "Sobreviventes x não sobreviventes", "sobrevivencia_barras",
               rotulos={0: "Não sobreviveu", 1: "Sobreviveu"})
# A interpretação desse resultado está no relatório (seção 3.3).

# %% 10. LOCAL DE EMBARQUE
tabela_frequencia(df_tratado["Embarked"], "Local de embarque")
grafico_barras(df_tratado["Embarked"], "Local de embarque", "embarque_barras",
               rotulos={"S": "Southampton", "C": "Cherbourg", "Q": "Queenstown"})
# A interpretação desse resultado está no relatório (seção 3.4).


# ======================================================================
# ETAPA 3 — VARIÁVEIS QUANTITATIVAS
# ======================================================================

def resumo_quantitativo(coluna, rotulo=None):
    """Calcula todas as medidas pedidas no roteiro para uma variável quantitativa."""
    dados = coluna.dropna()
    moda = dados.mode()
    resumo = {
        "Mínimo": dados.min(),
        "Máximo": dados.max(),
        "Média": dados.mean(),
        "Mediana": dados.median(),
        "Moda": moda.iloc[0] if not moda.empty else np.nan,
        "Variância": dados.var(),
        "Desvio-padrão": dados.std(),
        "Q1 (25%)": dados.quantile(0.25),
        "Q2 (50%)": dados.quantile(0.50),
        "Q3 (75%)": dados.quantile(0.75),
        "Amplitude": dados.max() - dados.min(),
        "Amplitude interquartílica (IQR)": dados.quantile(0.75) - dados.quantile(0.25),
    }
    tabela = pd.Series(resumo).round(2)
    print(f"\n--- Resumo estatístico: {rotulo or coluna.name} ---")
    print(tabela)
    return tabela


def grafico_hist_box(coluna, titulo, nome_arquivo):
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    sns.histplot(coluna.dropna(), kde=True, ax=axes[0], color="steelblue")
    axes[0].set_title(f"Histograma — {titulo}")
    sns.boxplot(x=coluna.dropna(), ax=axes[1], color="lightcoral")
    axes[1].set_title(f"Boxplot — {titulo}")
    fig.tight_layout()
    salvar(fig, nome_arquivo)


# %% 11. IDADE
resumo_quantitativo(df_tratado["Age"], "Idade")
grafico_hist_box(df_tratado["Age"], "Idade", "idade_hist_boxplot")
# A interpretação desse resultado está no relatório (seção 4.1).

# %% 12. TARIFA (Fare)
resumo_quantitativo(df_tratado["Fare"], "Tarifa (Fare)")
grafico_hist_box(df_tratado["Fare"], "Tarifa", "tarifa_hist_boxplot")
# A interpretação desse resultado está no relatório (seção 4.2).

# %% 13. DUAS VARIÁVEIS QUANTITATIVAS ADICIONAIS (SibSp e Parch)
resumo_quantitativo(df_tratado["SibSp"], "SibSp (irmãos/cônjuges a bordo)")
grafico_hist_box(df_tratado["SibSp"], "SibSp", "sibsp_hist_boxplot")

resumo_quantitativo(df_tratado["Parch"], "Parch (pais/filhos a bordo)")
grafico_hist_box(df_tratado["Parch"], "Parch", "parch_hist_boxplot")


# ======================================================================
# ETAPA 4 — COMPARAÇÃO ENTRE GRUPOS
# ======================================================================

def boxplot_comparativo(df_, x, y, titulo, nome_arquivo, rotulos=None):
    fig, ax = plt.subplots(figsize=(6, 4))
    dados_plot = df_.copy()
    if rotulos:
        dados_plot[x] = dados_plot[x].map(rotulos).fillna(dados_plot[x])
    sns.boxplot(data=dados_plot, x=x, y=y, hue=x, ax=ax, palette="Set2", legend=False)
    ax.set_title(titulo)
    salvar(fig, nome_arquivo)


# %% 14. COMPARAÇÃO 1 — IDADE: HOMENS x MULHERES
print("\n--- Idade por sexo ---")
print(df_tratado.groupby("Sex")["Age"].describe().round(2))
boxplot_comparativo(df_tratado, "Sex", "Age", "Idade: Homens x Mulheres",
                     "idade_por_sexo", rotulos={"male": "Homem", "female": "Mulher"})

# %% 15. COMPARAÇÃO 2 — IDADE: SOBREVIVENTES x NÃO SOBREVIVENTES
print("\n--- Idade por sobrevivência ---")
print(df_tratado.groupby("Survived")["Age"].describe().round(2))
boxplot_comparativo(df_tratado, "Survived", "Age", "Idade: Sobreviventes x Não sobreviventes",
                     "idade_por_sobrevivencia", rotulos={0: "Não sobreviveu", 1: "Sobreviveu"})

# %% 16. COMPARAÇÃO 3 — TARIFA POR CLASSE
print("\n--- Tarifa por classe ---")
print(df_tratado.groupby("Pclass")["Fare"].describe().round(2))
boxplot_comparativo(df_tratado, "Pclass", "Fare", "Tarifa por classe",
                     "tarifa_por_classe", rotulos={1: "1ª classe", 2: "2ª classe", 3: "3ª classe"})

# %% 17. EXTRA — SOBREVIVÊNCIA POR SEXO (percentual, útil para comparação 4 do roteiro)
tab_sexo_sobrev = pd.crosstab(df_tratado["Sex"], df_tratado["Survived"], normalize="index") * 100
print("\n--- Taxa de sobrevivência por sexo (%) ---")
print(tab_sexo_sobrev.round(2))

fig, ax = plt.subplots(figsize=(6, 4))
tab_sexo_sobrev.rename(columns={0: "Não sobreviveu", 1: "Sobreviveu"}).plot(
    kind="bar", stacked=True, ax=ax, colormap="Set2"
)
ax.set_title("Taxa de sobrevivência por sexo")
ax.set_ylabel("Percentual (%)")
ax.set_xlabel("")
ax.legend(title="")
plt.xticks(rotation=0)
salvar(fig, "sobrevivencia_por_sexo")
# A interpretação de todas as comparações acima está no relatório (seção 5).


# ======================================================================
# ETAPA 5 — INVESTIGAÇÃO LIVRE
# ======================================================================
# Pergunta: existe relação entre o título de tratamento presente no nome
# do passageiro (Mr., Mrs., Miss, Master) e a taxa de sobrevivência?
#
# Método: o título é extraído do campo Name por expressão regular, os
# títulos mais raros são agrupados em "Outro", e calcula-se a taxa de
# sobrevivência de cada grupo por meio de uma tabela cruzada.

# %% 18. INVESTIGAÇÃO LIVRE — TÍTULO DE TRATAMENTO E SOBREVIVÊNCIA
df_tratado["Titulo"] = df_tratado["Name"].str.extract(r",\s*([^\.]+)\.")
titulos_comuns = df_tratado["Titulo"].value_counts()
print("\nTítulos extraídos do nome:")
print(titulos_comuns)

# Método: agrupamos títulos raros em "Outro" e calculamos a taxa de
# sobrevivência por título.
principais = ["Mr", "Mrs", "Miss", "Master"]
df_tratado["Titulo_agrupado"] = df_tratado["Titulo"].apply(
    lambda t: t if t in principais else "Outro"
)
tab_titulo = pd.crosstab(df_tratado["Titulo_agrupado"], df_tratado["Survived"], normalize="index") * 100
print("\n--- Taxa de sobrevivência por título (%) ---")
print(tab_titulo.round(2))

fig, ax = plt.subplots(figsize=(6, 4))
tab_titulo.rename(columns={0: "Não sobreviveu", 1: "Sobreviveu"}).plot(
    kind="bar", stacked=True, ax=ax, colormap="coolwarm"
)
ax.set_title("Taxa de sobrevivência por título extraído do nome")
ax.set_ylabel("Percentual (%)")
ax.set_xlabel("")
ax.legend(title="")
plt.xticks(rotation=0)
salvar(fig, "sobrevivencia_por_titulo")
# Conclusão: o título reflete sexo e estado civil/idade ao mesmo tempo, e a
# taxa de sobrevivência segue a ordem esperada pela prioridade de evacuação
# (mulheres e crianças primeiro) — ver detalhamento no relatório (seção 6).


print("\n\nConcluído! Todos os gráficos estão na pasta graficos/ "
      "e a base tratada em titanic_tratado.csv.")
