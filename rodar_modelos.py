

import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
from scipy import stats
from statsmodels.stats.diagnostic import acorr_breusch_godfrey, het_breuschpagan, recursive_olsresiduals
from statsmodels.stats.stattools import jarque_bera, durbin_watson

ARQUIVO = "variaveis_modelo.xlsx"
CHOQUE = 10                     
FIM_TREINO = "2022-10-01"       
warnings.filterwarnings("ignore", category=FutureWarning)   

# ---------------------------------------------------------------------------
# PASSO 1 — Ler as variáveis e separar a amostra 
# ---------------------------------------------------------------------------
v = pd.read_excel(ARQUIVO, sheet_name="variaveis")
v["trimestre"] = pd.to_datetime(v["trimestre"])
amostra = v[v["na_amostra"] == 1].reset_index(drop=True)
amostra["d_2015T1"] = (amostra["rotulo"] == "2015T1").astype(int)   

L = range(1, 5)
BASE = ([f"inflacao_l{k}" for k in L] + [f"desemprego_l{k}" for k in L]
        + [f"selic_l{k}" for k in L] + ["d_t2", "d_t3", "d_t4"])
OLEO = {
    "Benchmark": [f"oleo_bench_l{k}" for k in range(5)],
    "Hamilton": [f"oleo_hamilton_l{k}" for k in range(5)],
    "Assimétrico": [f"oleo_alta_l{k}" for k in range(5)] + [f"oleo_queda_l{k}" for k in range(5)],
}


def estimar(dados, colunas, robusto=False):
    """Regressão por MQO: inflação explicada pelas colunas escolhidas."""
    X = sm.add_constant(dados[colunas], has_constant="add")
    res = sm.OLS(dados["inflacao"], X).fit()
    return res.get_robustcov_results("HC1") if robusto else res


def soma(res, colunas, sinal=1, menos=()):
    """Soma dos coeficientes (efeito em 1 ano) com erro-padrão, p-valor e intervalo de confiança de 95%.
    Com 'menos', calcula uma diferença: soma das 'colunas' menos soma das colunas em 'menos'."""
    t = res.t_test(" + ".join(colunas) + "".join(f" - {c}" for c in menos) + " = 0")
    s, ep, p = float(np.squeeze(t.effect)), float(np.squeeze(t.sd)), float(np.squeeze(t.pvalue))
    tc = stats.t.ppf(0.975, res.df_resid)   # valor crítico da distribuição t, a mesma do p-valor (com 70 gl ≈ 1,99)
    efeito = sinal * CHOQUE * s
    return {"efeito": efeito, "ep": CHOQUE * ep, "p": p,
            "ic_inf": efeito - tc * CHOQUE * ep, "ic_sup": efeito + tc * CHOQUE * ep}


# ---------------------------------------------------------------------------
# PASSO 2 — Estimar os três modelos 
# ---------------------------------------------------------------------------
modelos = {nome: estimar(amostra, BASE + oleo) for nome, oleo in OLEO.items()}


# ---------------------------------------------------------------------------
# PASSO 3 — Tabela no formato da Tabela 1 do artigo 
# ---------------------------------------------------------------------------
def rotulo(col):
    nomes = {"const": "Constante", "d_t2": "Dummy 2º trimestre", "d_t3": "Dummy 3º trimestre", "d_t4": "Dummy 4º trimestre"}
    if col in nomes:
        return nomes[col]
    base, k = col.rsplit("_l", 1)
    k = int(k)
    nome = {"inflacao": "Inflação", "desemprego": "Desemprego", "selic": "Selic",
            "oleo_bench": "Petróleo", "oleo_hamilton": "Petróleo", "oleo_alta": "Petróleo",
            "oleo_queda": "Petróleo – queda"}[base]
    return nome if k == 0 else f"{nome}(-{k})"


def br(x, casas=4):
    return f"{x:.{casas}f}".replace(".", ",")


def estrelas(p):
    return "***" if p < 0.01 else "**" if p < 0.05 else "*" if p < 0.10 else ""


ordem = (["const"] + [f"inflacao_l{k}" for k in L] + [f"desemprego_l{k}" for k in L] + [f"selic_l{k}" for k in L]
         + [f"oleo_bench_l{k}" for k in range(5)] + [f"oleo_queda_l{k}" for k in range(5)] + ["d_t2", "d_t3", "d_t4"])
linhas = []
for col in ordem:
    lin_coef, lin_ep = {"Variável": rotulo(col)}, {"Variável": ""}
    for nome, res in modelos.items():
        c = col
        if col.startswith("oleo_bench"):
            c = col.replace("oleo_bench", {"Benchmark": "oleo_bench", "Hamilton": "oleo_hamilton", "Assimétrico": "oleo_alta"}[nome])
        if c in res.params.index:
            lin_coef[nome] = br(res.params[c]) + estrelas(res.pvalues[c])
            lin_ep[nome] = "(" + br(res.bse[c]) + ")"
        else:
            lin_coef[nome], lin_ep[nome] = "", ""
    linhas += [lin_coef, lin_ep]


def medidas(res):
    n, k = res.nobs, res.df_model + 1
    y = res.model.endog
    return {"R²": res.rsquared, "R² ajustado": res.rsquared_adj, "Erro-padrão da regressão": np.sqrt(res.mse_resid),
            "Soma dos resíduos ao quadrado": res.ssr, "Log-verossimilhança": res.llf,
            "Durbin-Watson": durbin_watson(res.resid), "Média da variável dependente": y.mean(),
            "Desvio-padrão da variável dependente": y.std(ddof=1),
            "Critério de Akaike": -2 * res.llf / n + 2 * k / n,             
            "Critério de Schwarz": -2 * res.llf / n + k * np.log(n) / n,
            "Estatística F": res.fvalue, "Observações": n}


for medida in medidas(modelos["Benchmark"]):
    lin = {"Variável": medida}
    for nome, res in modelos.items():
        val = medidas(res)[medida]
        lin[nome] = str(int(val)) if medida == "Observações" else br(val)
    linhas.append(lin)
linhas.append({"Variável": "Notas: erros-padrão entre parênteses. *** p<0,01; ** p<0,05; * p<0,10. "
                           "No Assimétrico, as linhas 'Petróleo' são as altas.", "Benchmark": "", "Hamilton": "", "Assimétrico": ""})
tabela1 = pd.DataFrame(linhas)


# ---------------------------------------------------------------------------
# PASSO 4 — Efeito de uma alta de 10% do petróleo sobre o IPCA em um ano
# ---------------------------------------------------------------------------

efeitos = []
casos = [("Benchmark", "Alta de 10%", OLEO["Benchmark"], 1),
         ("Hamilton", "Alta de 10% acima do pico do último ano", OLEO["Hamilton"], 1),
         ("Assimétrico", "Alta de 10%", OLEO["Assimétrico"][:5], 1),
         ("Assimétrico", "Queda de 10%", OLEO["Assimétrico"][5:], -1)]
for nome, choque, cols, sinal in casos:
    res = modelos[nome]
    e = soma(res, cols, sinal)
    efeitos.append({"Modelo": nome, "Choque": choque,
                    "Efeito no IPCA acumulado em 1 ano (p.p.)": e["efeito"],
                    "IC 95% inferior": e["ic_inf"], "IC 95% superior": e["ic_sup"],
                    "p-valor (efeito = 0)": e["p"],
                    "Parcela no próprio trimestre (%)": 100 * res.params[cols[0]] / (sinal * e["efeito"] / CHOQUE) if sinal == 1 else np.nan,
                    "Na unidade do artigo (inflação anualizada, ×4)": 4 * e["efeito"]})
efeitos = pd.DataFrame(efeitos)
teste_assim = modelos["Assimétrico"].t_test(" + ".join(OLEO["Assimétrico"][:5]) + " - " + " - ".join(OLEO["Assimétrico"][5:]) + " = 0")
p_assimetria = float(np.squeeze(teste_assim.pvalue))


# ---------------------------------------------------------------------------
# PASSO 5 — Testes de diagnóstico (como no artigo): LM(2), Jarque-Bera, White e CUSUM
# ---------------------------------------------------------------------------
def white_sem_termos_cruzados(res):
    """Teste de White sem termos cruzados: n × R² da regressão dos resíduos² nas variáveis do modelo e nos
    seus quadrados. A conta é feita pela função het_breuschpagan da statsmodels, que recebe essas colunas."""
    X = res.model.exog
    aux = [np.ones(len(X))]
    for j in range(1, X.shape[1]):
        aux.append(X[:, j])
        if len(np.unique(X[:, j])) > 2:          
            aux.append(X[:, j] ** 2)
    lm, p, _, _ = het_breuschpagan(res.resid, np.column_stack(aux))
    return lm, p


def cusum(res):
    r = recursive_olsresiduals(res, alpha=0.95)
    caminho, banda = np.asarray(r[5]), np.asarray(r[6])
    caminho = caminho[-banda.shape[1]:]
    fora = bool(((caminho < banda[0]) | (caminho > banda[1])).any())
    return caminho, banda, fora


diag, curvas = [], {}
for nome, res in modelos.items():
    lm, lm_p, _, _ = acorr_breusch_godfrey(res, nlags=2)
    jb, jb_p, _, _ = jarque_bera(res.resid)
    wh, wh_p = white_sem_termos_cruzados(res)
    caminho, banda, fora = cusum(res)
    curvas[nome] = (caminho, banda)
    diag.append({"Modelo": nome,
                 "LM(2) autocorrelação — p-valor": lm_p, "Jarque-Bera normalidade — p-valor": jb_p,
                 "White heterocedasticidade — p-valor": wh_p,
                 "CUSUM": "sai das bandas de 5%" if fora else "fica dentro das bandas de 5%"})
diag = pd.DataFrame(diag)


# ---------------------------------------------------------------------------
# PASSO 6 — Teste de regime da Petrobras 
# ---------------------------------------------------------------------------

dr = amostra.copy()
inter_ctrl, inter_ppi = [], []
for k in range(5):
    dr[f"bench_l{k}_x_controle"] = dr[f"oleo_bench_l{k}"] * dr["d_controle"]
    dr[f"bench_l{k}_x_ppi"] = dr[f"oleo_bench_l{k}"] * dr["d_ppi"]
    inter_ctrl.append(f"bench_l{k}_x_controle")
    inter_ppi.append(f"bench_l{k}_x_ppi")
res_reg = estimar(dr, BASE + OLEO["Benchmark"] + ["d_controle", "d_ppi"] + inter_ctrl + inter_ppi)
b = OLEO["Benchmark"]
regimes = []
for periodo, cols in [("2003–2010 (antes do controle)", b),
                      ("2011–2016 (controle de preços)", b + inter_ctrl),
                      ("2017–2025 (PPI)", b + inter_ppi)]:
    e = soma(res_reg, cols)
    regimes.append({"Período": periodo, "Efeito no IPCA acumulado em 1 ano (p.p.)": e["efeito"],
                    "IC 95% inferior": e["ic_inf"], "IC 95% superior": e["ic_sup"], "p-valor (efeito = 0)": e["p"]})
for periodo, cols in [("Diferença: controle − antes", inter_ctrl), ("Diferença: PPI − antes", inter_ppi)]:
    e = soma(res_reg, cols)
    regimes.append({"Período": periodo, "Efeito no IPCA acumulado em 1 ano (p.p.)": e["efeito"],
                    "IC 95% inferior": e["ic_inf"], "IC 95% superior": e["ic_sup"], "p-valor (efeito = 0)": e["p"]})

e = soma(res_reg, inter_ppi, menos=inter_ctrl)
regimes.append({"Período": "Teste da H3 — diferença: PPI − controle", "Efeito no IPCA acumulado em 1 ano (p.p.)": e["efeito"],
                "IC 95% inferior": e["ic_inf"], "IC 95% superior": e["ic_sup"], "p-valor (efeito = 0)": e["p"]})

f = res_reg.f_test([" + ".join(inter_ctrl) + " = 0", " + ".join(inter_ppi) + " = 0"])
regimes.append({"Período": "Teste F: o efeito é o mesmo nos três períodos?",
                "p-valor (efeito = 0)": float(np.squeeze(f.pvalue))})


f_todas = res_reg.f_test(", ".join(f"{c} = 0" for c in inter_ctrl + inter_ppi))
regimes.append({"Período": "Complemento: as 10 interações são todas zero?",
                "p-valor (efeito = 0)": float(np.squeeze(f_todas.pvalue))})
regimes = pd.DataFrame(regimes)


# ---------------------------------------------------------------------------
# PASSO 7 — Robustez do Benchmark
# ---------------------------------------------------------------------------
rob = []
for desc, dados, cols, robusto in [
    ("Benchmark principal", amostra, BASE + b, False),
    ("Com dummies de 2015T1 e 2022T3 (os dois maiores resíduos)", amostra, BASE + b + ["d_2015T1", "d_2022T3"], False),
    ("Com dummies de 2020T2 e 2022T3 (pandemia e corte do ICMS)", amostra, BASE + b + ["d_2020T2", "d_2022T3"], False),
    ("Selic em variação (como no artigo)", amostra, [c.replace("selic_l", "selic_var_l") for c in BASE] + b, False),
    ("Erros-padrão robustos (HC1)", amostra, BASE + b, True),
]:
    res = estimar(dados, cols, robusto)
    e = soma(res, b)
    rob.append({"Especificação": desc, "Efeito no IPCA acumulado em 1 ano (p.p.)": e["efeito"],
                "IC 95% inferior": e["ic_inf"], "IC 95% superior": e["ic_sup"],
                "p-valor (efeito = 0)": e["p"], "R² ajustado": res.rsquared_adj,
                "Jarque-Bera — p-valor": jarque_bera(res.resid)[1]})
rob = pd.DataFrame(rob)


# ---------------------------------------------------------------------------
# PASSO 8 — Regressões recursivas 
# ---------------------------------------------------------------------------
rec = []
for m in range(44, len(amostra) + 1):
    e = soma(estimar(amostra.iloc[:m], BASE + b), b)
    rec.append({"Último trimestre usado": amostra["rotulo"].iloc[m - 1], "trimestre": amostra["trimestre"].iloc[m - 1],
                "Efeito no IPCA acumulado em 1 ano (p.p.)": e["efeito"], "IC 95% inferior": e["ic_inf"], "IC 95% superior": e["ic_sup"]})
rec = pd.DataFrame(rec)


# ---------------------------------------------------------------------------
# PASSO 9 — Previsão fora da amostra 
# ---------------------------------------------------------------------------
treino = amostra[amostra["trimestre"] <= FIM_TREINO]
teste = amostra[amostra["trimestre"] > FIM_TREINO]
prev = pd.DataFrame({"Trimestre": teste["rotulo"].values, "IPCA observado": teste["inflacao"].values})
erros = []
for nome, oleo in OLEO.items():
    cols = BASE + oleo
    res = estimar(treino, cols)
    f_ = res.predict(sm.add_constant(teste[cols], has_constant="add"))
    prev[nome] = f_.values
    a = teste["inflacao"].values
    erros.append({"Modelo": nome, "RMSE": np.sqrt(np.mean((f_.values - a) ** 2)), "MAE": np.mean(np.abs(f_.values - a)),
                  "MAPE (%)": 100 * np.mean(np.abs((f_.values - a) / a)),
                  "Theil U": np.sqrt(np.mean((f_.values - a) ** 2)) / (np.sqrt(np.mean(f_.values ** 2)) + np.sqrt(np.mean(a ** 2)))})
erros = pd.DataFrame(erros)


# ---------------------------------------------------------------------------
# PASSO 10 — Salvar a planilha
# ---------------------------------------------------------------------------
leia = pd.DataFrame([
    ("Tabela1", "Coeficientes dos três modelos, no formato da Tabela 1 de LeBlanc e Chinn (2004). Números com vírgula, prontos para o Word."),
    ("Efeito_1_ano", "Efeito de uma alta (ou queda) de 10% no petróleo sobre o IPCA acumulado em 12 meses, em pontos percentuais. "
                     "IC 95% calculado com a distribuição t, a mesma do p-valor. Para comparar com o artigo (EUA: 0,7 no Benchmark, "
                     "1,1 no Hamilton e 0,8 no Assimétrico), use a última coluna (×4), porque o artigo usa inflação anualizada."),
    ("Diagnosticos", "Testes do artigo. p-valor acima de 0,05 = o modelo passa no teste. CUSUM dentro das bandas = sem instabilidade detectada."),
    ("Regimes", "Efeito do petróleo antes, durante e depois do controle de preços da Petrobras, numa regressão só. Embaixo: "
                "o teste da H3 (efeito de 2017–2025 menos o de 2011–2016), o teste F da seção 3.4 (o efeito é o mesmo nos três "
                "períodos?) e, como complemento que não entra no texto, o teste de que as 10 interações são todas zero."),
    ("Robustez", "O efeito do Benchmark muda com dummies de trimestres atípicos (os dois maiores resíduos; ou pandemia e "
                 "corte do ICMS), com a Selic em variação ou com erros-padrão robustos?"),
    ("Recursivas", "Efeito estimado acrescentando um trimestre por vez (como a Figura 6 do artigo)."),
    ("Previsao", "Previsão estática (um trimestre à frente, com os valores observados das variáveis explicativas) para "
                 "2023T1–2025T4, com os modelos estimados até 2022T4. Embaixo, os erros de previsão usados no artigo: "
                 "RMSE, MAE, MAPE e Theil U (quanto menor, melhor)."),
], columns=["Aba", "O que tem"])
leia.loc[len(leia)] = ["Assimetria", "p-valor do teste 'efeito das altas = efeito das quedas': " + f"{p_assimetria:.3f}".replace(".", ",")]

with pd.ExcelWriter("resultados_modelos.xlsx", engine="openpyxl") as w:
    leia.to_excel(w, sheet_name="Leia-me", index=False)
    tabela1.to_excel(w, sheet_name="Tabela1", index=False)
    efeitos.round(3).to_excel(w, sheet_name="Efeito_1_ano", index=False)
    diag.round(3).to_excel(w, sheet_name="Diagnosticos", index=False)
    regimes.round(3).to_excel(w, sheet_name="Regimes", index=False)
    rob.round(3).to_excel(w, sheet_name="Robustez", index=False)
    rec.drop(columns="trimestre").round(3).to_excel(w, sheet_name="Recursivas", index=False)
    prev.round(3).to_excel(w, sheet_name="Previsao", index=False)
    erros.round(3).to_excel(w, sheet_name="Previsao", index=False, startrow=len(prev) + 2)
    for aba in w.sheets.values():
        for col in aba.columns:
            aba.column_dimensions[col[0].column_letter].width = 26 if col[0].column_letter != "A" else 34


# ---------------------------------------------------------------------------
# PASSO 11 — Figuras
# ---------------------------------------------------------------------------
AZUL, LARANJA, VERDE_AGUA = "#2a78d6", "#eb6834", "#1baf7a"
TEXTO, TEXTO_2, GRADE = "#0b0b0b", "#52514e", "#e6e5e1"
plt.rcParams.update({"font.size": 9, "axes.edgecolor": GRADE, "axes.labelcolor": TEXTO_2,
                     "xtick.color": TEXTO_2, "ytick.color": TEXTO_2})


def limpar(eixo):
    eixo.grid(axis="y", color=GRADE, linewidth=0.6)
    for lado in ["top", "right"]:
        eixo.spines[lado].set_visible(False)


# CUSUM: três painéis
fig, eixos = plt.subplots(1, 3, figsize=(12, 3.6), sharey=True)
for eixo, (nome, (caminho, banda)) in zip(eixos, curvas.items()):
    datas = amostra["trimestre"].iloc[-len(caminho):]
    eixo.plot(datas, caminho, color=AZUL, linewidth=2, label="CUSUM")
    eixo.plot(datas, banda[0], color=TEXTO_2, linewidth=1, linestyle="--", label="bandas de 5%")
    eixo.plot(datas, banda[1], color=TEXTO_2, linewidth=1, linestyle="--")
    eixo.set_title(nome, loc="left", color=TEXTO, fontsize=10)
    limpar(eixo)
eixos[0].legend(frameon=False, fontsize=8, loc="upper left")
fig.tight_layout()
fig.savefig("figura_cusum.png", dpi=200)

# Recursivas
fig, eixo = plt.subplots(figsize=(8, 3.8))
eixo.fill_between(rec["trimestre"], rec["IC 95% inferior"], rec["IC 95% superior"], color=AZUL, alpha=0.15, linewidth=0,
                  label="intervalo de confiança de 95%")
eixo.plot(rec["trimestre"], rec["Efeito no IPCA acumulado em 1 ano (p.p.)"], color=AZUL, linewidth=2,
          label="efeito de uma alta de 10% do petróleo (Benchmark)")
eixo.axhline(0, color=TEXTO_2, linewidth=0.8)
eixo.set_ylabel("p.p. no IPCA em 12 meses")
eixo.legend(frameon=False, fontsize=8, loc="lower right")
limpar(eixo)                     
fig.tight_layout()
fig.savefig("figura_recursivas.png", dpi=200)

# Previsão
fig, eixo = plt.subplots(figsize=(8, 3.8))
x = np.arange(len(prev))
eixo.plot(x, prev["IPCA observado"], color=TEXTO, linewidth=2, marker="o", markersize=5, label="IPCA observado")
for nome, cor in zip(OLEO, [AZUL, LARANJA, VERDE_AGUA]):
    eixo.plot(x, prev[nome], color=cor, linewidth=2, marker="o", markersize=4, label=nome)
eixo.set_xticks(x, prev["Trimestre"], rotation=45)
eixo.set_ylabel("IPCA, % no trimestre")
eixo.legend(frameon=False, ncol=4, loc="upper right")
limpar(eixo)
fig.tight_layout()
fig.savefig("figura_previsao.png", dpi=200)


# ---------------------------------------------------------------------------
# PASSO 12 — Resumo na tela
# ---------------------------------------------------------------------------
print("Trimestres na amostra:", int(modelos["Benchmark"].nobs))
print("\nEfeito de 10% no petróleo sobre o IPCA acumulado em 12 meses (p.p.):")
print(efeitos[["Modelo", "Choque", "Efeito no IPCA acumulado em 1 ano (p.p.)", "p-valor (efeito = 0)"]].round(3).to_string(index=False))
print(f"\nTeste de assimetria (altas = quedas): p-valor {p_assimetria:.3f}")
print("\nRegimes da Petrobras (Benchmark):")
print(regimes[["Período", "Efeito no IPCA acumulado em 1 ano (p.p.)", "p-valor (efeito = 0)"]].round(3).to_string(index=False))
print("\nArquivos gerados: resultados_modelos.xlsx, figura_cusum.png, figura_recursivas.png, figura_previsao.png")
