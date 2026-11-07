"""
Monta as variáveis do modelo (Curva de Phillips aumentada, LeBlanc e Chinn, 2004)
a partir da base congelada do TCC.

Como rodar:
    1. Deixe este arquivo na mesma pasta de base_modelo_final.xlsx
    2. No terminal, dentro dessa pasta:  python montar_variaveis.py
       (ou copie o código para uma célula do Jupyter/Colab e rode)

O que ele gera, na mesma pasta:
    - variaveis_modelo.xlsx         -> todas as variáveis, prontas para a regressão
    - figura_choques_petroleo.png   -> as 3 medidas de choque do petróleo (como a Figura 4 do artigo)

Precisa de: pandas, numpy, matplotlib e openpyxl.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

ARQUIVO_BASE = "base_modelo_final.xlsx"
DEFASAGENS = 4                      # 4 trimestres = 1 ano, como no artigo
INICIO_AMOSTRA = "2003-04-01"       # 2003T2: primeiro trimestre com as 4 defasagens do desemprego


# ---------------------------------------------------------------------------
# PASSO 1 — Ler a base congelada
# ---------------------------------------------------------------------------
base = pd.read_excel(ARQUIVO_BASE, sheet_name="base_modelo")
base["trimestre"] = pd.to_datetime(base["trimestre"])
base = base.sort_values("trimestre").reset_index(drop=True)

v = pd.DataFrame()
v["trimestre"] = base["trimestre"]
v["rotulo"] = base["rotulo"]


# ---------------------------------------------------------------------------
# PASSO 2 — Variáveis da Curva de Phillips (equação 1 do TCC)
# ---------------------------------------------------------------------------
v["inflacao"] = base["ipca_pct"]            # π: IPCA, % no trimestre
v["desemprego"] = base["desemprego_pct"]    # U: taxa de desemprego, %
v["selic"] = base["selic_pct"]              # i: Selic, % ao ano (em nível)

# Alternativa para os juros: LeBlanc e Chinn usam a VARIAÇÃO (log) da taxa de juros.
# Fica pronta aqui para testar depois; o modelo principal decide qual usar.
v["selic_var"] = 100 * np.log(base["selic_pct"] / base["selic_pct"].shift(1))


# ---------------------------------------------------------------------------
# PASSO 3 — As três medidas de choque do petróleo (Brent em R$)
# ---------------------------------------------------------------------------
preco = base["brent_brl"]

# Modelo 1 — Benchmark (equação 2): variação % do preço em relação ao trimestre anterior
v["oleo_bench"] = 100 * (preco / preco.shift(1) - 1)

# Modelo 2 — Net price de Hamilton (equação 3):
# compara o preço de hoje com o MAIOR preço dos 4 trimestres anteriores.
# Se superou, guarda quanto superou (em %); se não superou, vale zero.
maximo_4_trim = preco.shift(1).rolling(4).max()
v["oleo_hamilton"] = 100 * np.maximum(0, preco / maximo_4_trim - 1)
v.loc[maximo_4_trim.isna(), "oleo_hamilton"] = np.nan   # sem 4 trimestres de histórico, não dá para calcular

# Modelo 3 — Assimétrico (equações 4 e 5): separa altas e quedas em duas colunas
v["oleo_alta"] = v["oleo_bench"].clip(lower=0)     # só as altas; quedas viram 0
v["oleo_queda"] = v["oleo_bench"].clip(upper=0)    # só as quedas; altas viram 0


# ---------------------------------------------------------------------------
# PASSO 4 — Defasagens (lags): o valor de 1, 2, 3 e 4 trimestres atrás
# ---------------------------------------------------------------------------
# Inflação, desemprego e juros entram só defasados (t-1 a t-4), como na Tabela 1 do artigo.
for nome in ["inflacao", "desemprego", "selic", "selic_var"]:
    for k in range(1, DEFASAGENS + 1):
        v[f"{nome}_l{k}"] = v[nome].shift(k)

# O petróleo entra no trimestre atual (l0) e nos 4 anteriores (l1 a l4).
for nome in ["oleo_bench", "oleo_hamilton", "oleo_alta", "oleo_queda"]:
    v[f"{nome}_l0"] = v[nome]
    for k in range(1, DEFASAGENS + 1):
        v[f"{nome}_l{k}"] = v[nome].shift(k)


# ---------------------------------------------------------------------------
# PASSO 5 — Variáveis "liga/desliga" (dummies): valem 1 ou 0
# ---------------------------------------------------------------------------
tri = v["trimestre"].dt.quarter
v["d_t2"] = (tri == 2).astype(int)      # sazonalidade: o 1º trimestre fica como referência
v["d_t3"] = (tri == 3).astype(int)
v["d_t4"] = (tri == 4).astype(int)

ano = v["trimestre"].dt.year
v["d_controle"] = ((ano >= 2011) & (ano <= 2016)).astype(int)   # controle de preços da Petrobras
v["d_ppi"] = (ano >= 2017).astype(int)                          # Política de Paridade de Importação

v["d_2020T2"] = (v["rotulo"] == "2020T2").astype(int)   # pandemia
v["d_2022T3"] = (v["rotulo"] == "2022T3").astype(int)   # corte de ICMS sobre combustíveis


# ---------------------------------------------------------------------------
# PASSO 6 — Marcar a amostra de estimação
# ---------------------------------------------------------------------------
precisa = (["inflacao"]
           + [f"{n}_l{k}" for n in ["inflacao", "desemprego", "selic"] for k in range(1, DEFASAGENS + 1)]
           + [f"{n}_l{k}" for n in ["oleo_bench", "oleo_hamilton", "oleo_alta", "oleo_queda"] for k in range(0, DEFASAGENS + 1)])
v["na_amostra"] = ((v["trimestre"] >= INICIO_AMOSTRA) & v[precisa].notna().all(axis=1)).astype(int)


# ---------------------------------------------------------------------------
# PASSO 7 — Salvar a planilha com um dicionário das colunas
# ---------------------------------------------------------------------------
dicionario = pd.DataFrame([
    ("inflacao", "IPCA, % no trimestre (π)"),
    ("desemprego", "Taxa de desemprego, % (U). 2002–2011: PME compatibilizada; 2012+: PNAD Contínua"),
    ("selic", "Selic, % ao ano, em nível (i)"),
    ("selic_var", "Variação log da Selic × 100 (alternativa, como no artigo)"),
    ("oleo_bench", "Modelo Benchmark: variação % do Brent em R$ no trimestre"),
    ("oleo_hamilton", "Modelo de Hamilton: % acima do maior preço dos 4 trimestres anteriores (0 se não superou)"),
    ("oleo_alta / oleo_queda", "Modelo Assimétrico: variação % separada em altas e quedas"),
    ("_l0 ... _l4", "Mesma variável no trimestre atual (l0) e 1 a 4 trimestres atrás (l1 a l4)"),
    ("d_t2, d_t3, d_t4", "Dummies de sazonalidade (1º trimestre = referência)"),
    ("d_controle", "1 em 2011–2016 (controle de preços dos combustíveis)"),
    ("d_ppi", "1 de 2017 em diante (Política de Paridade de Importação)"),
    ("d_2020T2, d_2022T3", "Dummies de trimestres atípicos"),
    ("na_amostra", "1 = trimestre usado na regressão (2003T2–2025T4)"),
], columns=["coluna", "o que é"])

with pd.ExcelWriter("variaveis_modelo.xlsx") as arquivo:
    v.to_excel(arquivo, sheet_name="variaveis", index=False)
    dicionario.to_excel(arquivo, sheet_name="dicionario", index=False)


# ---------------------------------------------------------------------------
# PASSO 8 — Gráfico das três medidas de choque (como a Figura 4 do artigo)
# ---------------------------------------------------------------------------
periodo = v[v["trimestre"] >= "2002-01-01"]
paineis = [
    ("oleo_bench", "Variação % do Brent em R$ (Benchmark)"),
    ("oleo_hamilton", "Net price de Hamilton"),
    ("oleo_alta", "Altas do Brent em R$ (Assimétrico)"),
    ("oleo_queda", "Quedas do Brent em R$ (Assimétrico)"),
]
AZUL, TEXTO, TEXTO_2, GRADE, FAIXA = "#2a78d6", "#0b0b0b", "#52514e", "#e6e5e1", "#f1f0ec"

plt.rcParams.update({"font.size": 9, "axes.edgecolor": GRADE, "axes.labelcolor": TEXTO_2,
                     "xtick.color": TEXTO_2, "ytick.color": TEXTO_2})
fig, eixos = plt.subplots(2, 2, figsize=(10, 6), sharex=True)
for eixo, (coluna, titulo) in zip(eixos.flat, paineis):
    eixo.axvspan(pd.Timestamp("2011-01-01"), pd.Timestamp("2016-12-31"), color=FAIXA, zorder=0)
    eixo.bar(periodo["trimestre"], periodo[coluna], width=70, color=AZUL, zorder=2)
    eixo.axhline(0, color=TEXTO_2, linewidth=0.8, zorder=3)
    eixo.set_title(titulo, loc="left", color=TEXTO, fontsize=10)
    eixo.set_ylabel("%")
    eixo.grid(axis="y", color=GRADE, linewidth=0.6, zorder=1)
    for lado in ["top", "right"]:
        eixo.spines[lado].set_visible(False)
eixos[0, 0].text(pd.Timestamp("2011-03-01"), eixos[0, 0].get_ylim()[1] * 0.92,
                 "controle de preços\n2011–2016", fontsize=8, color=TEXTO_2, va="top")
fig.tight_layout()
fig.savefig("figura_choques_petroleo.png", dpi=200)


# ---------------------------------------------------------------------------
# PASSO 9 — Conferência: leia estas linhas depois de rodar
# ---------------------------------------------------------------------------
amostra = v[v["na_amostra"] == 1]
print("Trimestres na amostra:", len(amostra), "| de", amostra["rotulo"].iloc[0], "a", amostra["rotulo"].iloc[-1])
print("Altas + quedas = variação total em todos os trimestres?",
      bool(np.allclose((v["oleo_alta"] + v["oleo_queda"]).dropna(), v["oleo_bench"].dropna())))
print("Trimestres em que o net price de Hamilton foi positivo:",
      int((amostra["oleo_hamilton"] > 0).sum()), "de", len(amostra))
print("\nMaiores altas e quedas do Brent em R$ (Benchmark):")
extremos = pd.concat([v.nlargest(3, "oleo_bench"), v.nsmallest(3, "oleo_bench")])
print(extremos[["rotulo", "oleo_bench"]].round(1).to_string(index=False))
print("\nArquivos gerados: variaveis_modelo.xlsx e figura_choques_petroleo.png")
