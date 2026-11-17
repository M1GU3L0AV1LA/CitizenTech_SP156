# ==============================================================================
# CitizenTech | Análise estatística da Central SP156 (Prefeitura de São Paulo)
# Pergunta de negócio: quais fatores aumentam o tempo de atendimento de um
# serviço e o que um assistente digital deve priorizar?
#
# Fonte dos dados: Dados Abertos da Prefeitura de São Paulo (SP156), julho/2026
# Execução: python src/citizentech_sp156.py  (a partir de qualquer pasta)
# ==============================================================================

# Preparando o ambiente
from pathlib import Path
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from scipy import stats
from scipy.stats import norm
from sklearn.metrics import confusion_matrix, accuracy_score

# ------------------------------------------------------------------------------
# Configurações
# ------------------------------------------------------------------------------
SEED = 42                  # seed fixa: resultados reproduzíveis
MOSTRAR_GRAFICOS = True    # False = apenas salva os PNGs, sem abrir janelas
FONTE = "Fonte: Dados Abertos SP156 - Prefeitura de São Paulo (julho/2026)"

PASTA = Path(__file__).resolve().parent.parent
PASTA_FIG = PASTA / "figuras"
PASTA_TAB = PASTA / "tabelas"
PASTA_FIG.mkdir(exist_ok=True)
PASTA_TAB.mkdir(exist_ok=True)

pd.set_option("display.max_columns", None)
pd.set_option("display.width", None)


def finalizar_grafico(nome):
    """Adiciona a fonte, salva o PNG e mostra (ou fecha) o gráfico."""
    plt.gcf().text(0.99, 0.005, FONTE, ha="right", fontsize=7, color="gray")
    plt.savefig(PASTA_FIG / f"{nome}.png", dpi=150, bbox_inches="tight")
    if MOSTRAR_GRAFICOS:
        plt.show()
    else:
        plt.close()


def titulo(texto):
    print("\n" + "=" * 78)
    print(texto)
    print("=" * 78)


# ==============================================================================
# 1) LEITURA E PREPARAÇÃO DA BASE
# ==============================================================================
titulo("1) LEITURA E PREPARAÇÃO DA BASE")

bruta = pd.read_csv(PASTA / "dados" / "sp156.csv", low_memory=False)
print("Solicitações na base bruta:", len(bruta))

# As datas vêm no formato mês/dia/ano
df = bruta.copy()
df["abertura"] = pd.to_datetime(df["Data de Abertura"], format="%m/%d/%Y")
df["finalizacao"] = pd.to_datetime(df["Data de Finalização"],
                                   format="%m/%d/%Y", errors="coerce")
print("Período de abertura:", df["abertura"].min().date(),
      "até", df["abertura"].max().date())

# Tempo de atendimento (dias) = finalização - abertura
df["tempo_dias"] = (df["finalizacao"] - df["abertura"]).dt.days

# Critérios de inclusão (anotar as contagens no relatório)
df = df[df["Status"] == "Finalizada"]
print("Após Status = Finalizada:", len(df))
df = df.dropna(subset=["tempo_dias"])
print("Após exigir data de finalização:", len(df))
print("Tempos negativos:", int((df["tempo_dias"] < 0).sum()))

t = df["tempo_dias"]

# ==============================================================================
# 2) ESTATÍSTICA DESCRITIVA DO TEMPO DE ATENDIMENTO
# ==============================================================================
titulo("2) ESTATÍSTICA DESCRITIVA - tempo de atendimento (dias)")

# Medidas de Tendência Central:
print("Média:", round(t.mean(), 3))
print("Mediana:", t.median())
print("Moda:", t.mode().iloc[0])

# Medidas de Dispersão:
print("Mínimo:", t.min())
print("Máximo:", t.max())
print("Amplitude:", t.max() - t.min())
print("Variância:", round(t.var(), 3))
print("Desvio padrão:", round(t.std(), 3))
print("Coeficiente de variação (%):", round(t.std() / t.mean() * 100, 1))

# Medidas Separatrizes:
print("Quartis:\n", t.quantile([0.25, 0.50, 0.75]))
print("Percentis 90, 95 e 99:\n", t.quantile([0.90, 0.95, 0.99]))
print("Medidas Resumo:\n", t.describe())
print("% resolvidas no mesmo dia:", round((t == 0).mean() * 100, 1))

# Tabela de distribuição de frequências (variável contínua, regra de Sturges)
n_t = len(t)
k = int(round(1 + 3.322 * math.log10(n_t)))
classes = pd.cut(t, bins=k)
fi = classes.value_counts().sort_index()
fr = (fi / fi.sum() * 100).round(2)
tabela = pd.DataFrame({
    "Classe (dias)": [str(i) for i in fi.index],
    "fi": fi.values,
    "fr%": fr.values,
    "Fac": fi.cumsum().values,
    "Frc%": fr.cumsum().round(2).values,
})
print(f"\nTabela de frequências (Sturges, k = {k} classes):\n", tabela.to_string(index=False))
tabela.to_csv(PASTA_TAB / "frequencias_tempo.csv", index=False)

# Análise gráfica: histograma e boxplot
plt.figure(figsize=(8, 5))
plt.hist(t.clip(upper=t.quantile(0.99)), bins=40, color="steelblue", edgecolor="white")
plt.title("Distribuição do tempo de atendimento (SP156)")
plt.xlabel("Dias até a finalização (truncado no percentil 99)")
plt.ylabel("Nº de solicitações")
plt.grid(axis="y", alpha=0.3)
finalizar_grafico("hist_tempo_dias")

plt.figure(figsize=(6, 5))
plt.boxplot(t, patch_artist=True, boxprops=dict(facecolor="skyblue"))
plt.title("Boxplot - tempo de atendimento")
plt.ylabel("Dias")
finalizar_grafico("boxplot_tempo_dias")

# Verificação da suposição de normalidade
print("\nAssimetria:", round(stats.skew(t), 2), "| Curtose:", round(stats.kurtosis(t), 2))
amostra_sw = t.sample(5000, random_state=SEED)
print("Shapiro-Wilk (amostra de 5000):", stats.shapiro(amostra_sw))

sm.qqplot(t, line="s")
plt.title("QQ-plot do tempo de atendimento")
plt.xlabel("Quantis teóricos (Normal)")
plt.ylabel("Quantis da amostra (dias)")
finalizar_grafico("qqplot_tempo")

# ==============================================================================
# 3) ITEM 02 - PROBABILIDADE (supondo Distribuição Normal)
# ==============================================================================
titulo("3) ITEM 02 - PROBABILIDADE")

media = t.mean()
mediana = t.median()
desvio = t.std(ddof=0)   # desvio padrão populacional
print(f"Média = {media:.3f} | Mediana = {mediana:.1f} | Desvio padrão = {desvio:.3f}")


def classificar_evento(p):
    """Classificação clássica do evento quanto ao resultado probabilístico."""
    if np.isclose(p, 0):
        return "Impossível"
    if np.isclose(p, 1):
        return "Certo"
    if np.isclose(p, 0.5):
        return "Equiprovável"
    return "Improvável" if p < 0.5 else "Provável"


# a) P(X < média)
p_a = norm.cdf(media, media, desvio)
print(f"\na) P(X < média)   = {p_a:.4f} -> evento {classificar_evento(p_a)}")

# b) P(X > mediana)
p_b = norm.sf(mediana, media, desvio)
print(f"b) P(X > mediana) = {p_b:.4f} -> evento {classificar_evento(p_b)}")

# Observação para o relatório: a Normal atribuiria probabilidade a tempos < 0
print(f"\nP(X < 0) sob a Normal = {norm.cdf(0, media, desvio):.4f} (tempo negativo é impossível)")

# ==============================================================================
# 4) ITEM 03 - TESTE DE HIPÓTESE PARA A MÉDIA (teste Z, sigma conhecido)
# ==============================================================================
titulo("4) ITEM 03 - TESTE DE HIPÓTESE PARA A MÉDIA")

mu0 = media          # média populacional de referência
sigma = desvio       # desvio padrão populacional de referência


def teste_z(n, alfa, cauda, rotulo):
    amostra = t.sample(n, random_state=SEED)
    xbar = amostra.mean()
    ep = sigma / np.sqrt(n)            # erro padrão da média amostral
    z = (xbar - mu0) / ep
    if cauda == "esquerda":            # H1: mu < mu0
        zc = norm.ppf(alfa)
        p_valor = norm.cdf(z)
        rejeita = z < zc
        h1 = "mu < mu0"
    else:                              # H1: mu > mu0
        zc = norm.ppf(1 - alfa)
        p_valor = norm.sf(z)
        rejeita = z > zc
        h1 = "mu > mu0"
    print(f"\n{rotulo}) H0: mu = {mu0:.3f} | H1: {h1} | alfa = {alfa:.0%} | n = {n}")
    print(f"   Média amostral = {xbar:.3f} | erro padrão = {ep:.3f}")
    print(f"   Z calculado = {z:.4f} | Z crítico = {zc:.4f} | p-valor = {p_valor:.4f}")
    print("   Decisão:", "REJEITA H0" if rejeita else "NÃO REJEITA H0")


teste_z(n=20, alfa=0.06, cauda="esquerda", rotulo="a")
teste_z(n=25, alfa=0.09, cauda="direita", rotulo="b")

# ==============================================================================
# 5) BASE POR SERVIÇO (unidade de análise dos itens 04, 05 e 06)
# ==============================================================================
titulo("5) BASE POR SERVIÇO")

# Métricas por serviço, calculadas sobre as solicitações finalizadas
serv = df.groupby("Serviço").agg(
    n_solicitacoes=("tempo_dias", "size"),
    tempo_medio=("tempo_dias", "mean"),
    tempo_mediano=("tempo_dias", "median"),
    pct_fim_semana=("abertura", lambda s: (s.dt.dayofweek >= 5).mean() * 100),
    pct_com_endereco=("Latitude", lambda s: s.notna().mean() * 100),
)

# % de solicitações por canal de comunicação
canais = pd.crosstab(df["Serviço"], df["Meio de Comunicação"], normalize="index") * 100
canais.columns = ["pct_" + c.lower() for c in canais.columns]
serv = serv.join(canais)

# Só serviços com volume suficiente (medianas instáveis em serviços raros)
MIN_SOLICITACOES = 30
print("Serviços antes do filtro:", len(serv))
serv = serv[serv["n_solicitacoes"] >= MIN_SOLICITACOES]
print(f"Serviços com >= {MIN_SOLICITACOES} solicitações:", len(serv))
serv = serv.reset_index()

# Órgão principal de cada serviço
orgao_serv = df.groupby("Serviço")["Órgão"].agg(lambda s: s.mode().iloc[0])
serv["Órgão"] = serv["Serviço"].map(orgao_serv)

# Variáveis por órgão, calculadas na base bruta (todos os status)
org = bruta.groupby("Órgão").agg(
    carga_orgao=("Status", "size"),
    pct_nao_finalizada_orgao=("Status", lambda s: (s != "Finalizada").mean() * 100),
)
serv = serv.merge(org, left_on="Órgão", right_index=True, how="left")

# Transformação logarítmica do volume (reduz a assimetria)
serv["log_n"] = np.log10(serv["n_solicitacoes"])
serv["tempo_log"] = np.log1p(serv["tempo_mediano"])

serv.round(2).to_csv(PASTA / "dados" / "base_servicos_v2.csv", index=False)
print("Órgãos distintos:", serv["Órgão"].nunique())
print("\nMais lentos:\n", serv.nlargest(5, "tempo_mediano")[["Serviço", "n_solicitacoes", "tempo_mediano"]])
print("\nMais rápidos:\n", serv.nsmallest(5, "tempo_mediano")[["Serviço", "n_solicitacoes", "tempo_mediano"]])

# ==============================================================================
# 6) ITEM 04 - ANÁLISE BIVARIADA
#    Y = tempo mediano (dias) | X = % de solicitações não finalizadas no órgão
# ==============================================================================
titulo("6) ITEM 04 - ANÁLISE BIVARIADA")

X_NOME = "pct_nao_finalizada_orgao"
Y_NOME = "tempo_mediano"
dados4 = serv[[X_NOME, Y_NOME]]
print("n =", len(dados4))

# A) Gráfico de dispersão
sns.scatterplot(x=X_NOME, y=Y_NOME, data=dados4)
plt.title("Dispersão: fila do órgão x tempo mediano de atendimento")
plt.xlabel("% de solicitações não finalizadas no órgão")
plt.ylabel("Tempo mediano de atendimento (dias)")
finalizar_grafico("item04_dispersao")

# B) Covariância e Coeficiente de Correlação de Pearson
print("\nCovariância:", round(np.cov(dados4[X_NOME], dados4[Y_NOME])[0, 1], 3))
print("Correlação de Pearson:", round(np.corrcoef(dados4[X_NOME], dados4[Y_NOME])[0, 1], 4))

# C) Gráfico de correlação
sns.heatmap(dados4.corr(), annot=True, fmt=".3f", cmap="coolwarm", vmin=-1, vmax=1)
plt.title("Mapa de Correlação")
finalizar_grafico("item04_correlacao")

# D) Reta da regressão linear simples (no gráfico de dispersão) e equação
modelo4 = sm.OLS(dados4[Y_NOME], sm.add_constant(dados4[X_NOME])).fit()
intercepto, coeficiente = modelo4.params

plt.figure(figsize=(7, 5))
sns.regplot(x=X_NOME, y=Y_NOME, data=dados4, ci=None, line_kws={"color": "red"})
plt.title("Gráfico de Dispersão com Reta de Regressão")
plt.xlabel("% de solicitações não finalizadas no órgão")
plt.ylabel("Tempo mediano de atendimento (dias)")
finalizar_grafico("item04_regressao")

# Y = a + b.X + E
print(f"\nEquação: tempo_mediano = {intercepto:.3f} + {coeficiente:.3f} * {X_NOME}")

# E) Adequação do modelo: R² e Teste F
print(modelo4.summary())
print(f"R² = {modelo4.rsquared:.4f} | R² ajustado = {modelo4.rsquared_adj:.4f}")
print(f"Estatística F = {modelo4.fvalue:.3f} | p-valor do Teste F = {modelo4.f_pvalue:.4g}")

# F) Previsão (valor dentro do intervalo observado de X)
nova_obs = 30
print(f"\nIntervalo observado de X: {dados4[X_NOME].min():.1f} a {dados4[X_NOME].max():.1f}")
print(f"Previsão: órgão com {nova_obs}% não finalizadas -> "
      f"{intercepto + coeficiente * nova_obs:.1f} dias (tempo mediano)")

# ==============================================================================
# 7) ITEM 05 - ANÁLISE MULTIVARIADA (regressão linear múltipla)
#    Y = tempo mediano | X = pendências do órgão, % via portal, log10(volume)
# ==============================================================================
titulo("7) ITEM 05 - REGRESSÃO LINEAR MÚLTIPLA")

XS = ["pct_nao_finalizada_orgao", "pct_portal", "log_n"]
dados5 = serv[["tempo_mediano"] + XS]

# A) Gráfico de dispersão
sns.pairplot(dados5, kind="reg", diag_kind=None)
plt.suptitle("Dispersão entre as variáveis", y=1.02)
finalizar_grafico("item05_dispersao_matriz")

# B) Matriz de Covariância e Matriz de Correlação:
print("Matriz de covariância:\n", dados5.cov().round(2))
print("\nMatriz de correlação de Pearson:\n", dados5.corr().round(3))

# C) Gráfico de correlação
plt.figure(figsize=(7, 5.5))
sns.heatmap(dados5.corr(), annot=True, fmt=".2f", cmap="coolwarm", vmin=-1, vmax=1)
plt.title("Mapa de Correlação entre Variáveis")
finalizar_grafico("item05_correlacao")

# D) Equação da regressão linear múltipla
X5 = sm.add_constant(dados5[XS])
modelo5 = sm.OLS(dados5["tempo_mediano"], X5).fit()
b = modelo5.params
print(f"\nEquação: tempo_mediano = {b['const']:.3f} "
      f"{b['pct_nao_finalizada_orgao']:+.3f}*pct_nao_finalizada_orgao "
      f"{b['pct_portal']:+.3f}*pct_portal {b['log_n']:+.3f}*log_n")

# E) Adequação do modelo: R², R² ajustado e Teste F
print(modelo5.summary())
print(f"R² = {modelo5.rsquared:.4f} | R² ajustado = {modelo5.rsquared_adj:.4f}")
print(f"p-valor do Teste F = {modelo5.f_pvalue:.4g}")

# F) Previsão com 3 variáveis explicativas
print("\nFaixas observadas:\n", dados5[XS].describe().loc[["min", "50%", "max"]].round(2))
novo5 = pd.DataFrame({"const": [1.0], "pct_nao_finalizada_orgao": [30.0],
                      "pct_portal": [60.0], "log_n": [np.log10(500)]})
prev5 = modelo5.get_prediction(novo5).summary_frame(alpha=0.05)
print("\nPrevisão (30% pendentes, 60% portal, 500 solicitações):",
      round(float(prev5["mean"].iloc[0]), 2), "dias")

# ==============================================================================
# 8) ITEM 06 - REGRESSÃO LOGÍSTICA BINÁRIA
#    Y = 1 se o serviço é "lento" (tempo mediano acima da mediana dos serviços)
# ==============================================================================
titulo("8) ITEM 06 - REGRESSÃO LOGÍSTICA BINÁRIA")

CORTE = serv["tempo_mediano"].median()
dados6 = serv[XS].copy()
dados6["lento"] = (serv["tempo_mediano"] > CORTE).astype(int)
print(f"Corte (mediana dos serviços) = {CORTE:.1f} dias")
print("Distribuição da classe:\n", dados6["lento"].value_counts())
ref = max(dados6["lento"].mean(), 1 - dados6["lento"].mean())
print(f"Acurácia de referência (classe majoritária): {ref:.3f}")

# A) Modelo
X6 = sm.add_constant(dados6[XS])
modelo6 = sm.Logit(dados6["lento"], X6).fit(disp=False)

# B) Curvas sigmoides (cada variável varia; as demais ficam na média)
fig, eixos = plt.subplots(1, 3, figsize=(15, 4.5))
for ax, var in zip(eixos, XS):
    grade = np.linspace(dados6[var].min(), dados6[var].max(), 100)
    X_curva = pd.DataFrame({"const": 1.0, **{v: dados6[v].mean() for v in XS}}, index=range(100))
    X_curva[var] = grade
    ax.plot(grade, modelo6.predict(X_curva))
    ax.scatter(dados6[var], dados6["lento"], alpha=0.4)
    ax.set_title(f"Sigmoide - {var}")
    ax.set_xlabel(var)
    ax.set_ylabel("Probabilidade de serviço lento")
plt.tight_layout()
finalizar_grafico("item06_sigmoides")

# C) Equação (montar com base nos resultados)
print(modelo6.summary())
bl = modelo6.params
print(f"\nlogit(p) = ln(p/(1-p)) = {bl['const']:.3f} "
      f"{bl['pct_nao_finalizada_orgao']:+.3f}*pct_nao_finalizada_orgao "
      f"{bl['pct_portal']:+.3f}*pct_portal {bl['log_n']:+.3f}*log_n")
print("p = 1 / (1 + exp(-logit))")
odds = pd.DataFrame({"odds_ratio": np.exp(bl),
                     "IC95_inf": np.exp(modelo6.conf_int()[0]),
                     "IC95_sup": np.exp(modelo6.conf_int()[1])}).round(3)
print("\nOdds ratios:\n", odds)
print(f"Pseudo-R² (McFadden) = {modelo6.prsquared:.4f} | p (razão de verossimilhança) = {modelo6.llr_pvalue:.4g}")

# D) Previsões e classes
previsoes = modelo6.predict(X6)
classe = (previsoes > 0.5).astype(int)

# E) Matriz de confusão e acurácia
matriz_confusao = confusion_matrix(dados6["lento"], classe)
acuracia = accuracy_score(dados6["lento"], classe)
vn, fp, fn, vp = matriz_confusao.ravel()
print("\nMatriz de confusão (linhas = real, colunas = previsto):\n", matriz_confusao)
print(f"Acurácia = {acuracia:.3f}")
print(f"Sensibilidade = {vp/(vp+fn):.3f} | Especificidade = {vn/(vn+fp):.3f}")

sns.heatmap(matriz_confusao, annot=True, fmt="d", cmap="Blues",
            xticklabels=["Não lento", "Lento"], yticklabels=["Não lento", "Lento"])
plt.title("Matriz de confusão - regressão logística")
plt.xlabel("Classe prevista")
plt.ylabel("Classe real")
finalizar_grafico("item06_matriz_confusao")

# F) Classificação de cada serviço
dados6["classificacao"] = classe
print("\nClassificação (amostra):\n", dados6.head())

# G) Classificação de serviços novos, com as 3 variáveis explicativas
for pend, portal, volume in [(30, 60, 500), (2, 60, 500)]:
    novo = pd.DataFrame({"const": [1.0], "pct_nao_finalizada_orgao": [pend],
                         "pct_portal": [portal], "log_n": [np.log10(volume)]})
    prob = modelo6.predict(novo)
    classe_novo = (prob > 0.5).astype(int)
    print(f"Serviço novo ({pend}% pendentes, {portal}% portal, {volume} solic.): "
          f"Probabilidade = {prob.values[0]:.3f} | Classe prevista = {classe_novo.values[0]} "
          f"({'lento' if classe_novo.values[0] else 'não lento'})")

# ==============================================================================
# 9) EXTRAS (rigor adicional, além do conteúdo de aula)
# ==============================================================================
titulo("9) EXTRAS - robustez, validação e insights para o assistente digital")

from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, roc_curve

grupos = pd.factorize(serv["Órgão"])[0]   # serviços do mesmo órgão não são independentes

# 9.1 Multicolinearidade (VIF): regra prática, VIF > 5 pede atenção
vif = pd.DataFrame({"variavel": X5.columns,
                    "VIF": [variance_inflation_factor(X5.values, i) for i in range(X5.shape[1])]})
print("\nVIF:\n", vif[vif["variavel"] != "const"].round(2))

# 9.2 Erros-padrão agrupados por órgão (regressão múltipla e logística)
rob5 = sm.OLS(dados5["tempo_mediano"], X5).fit(cov_type="cluster", cov_kwds={"groups": grupos})
print("\nRegressão múltipla com erro-padrão por órgão:\n", rob5.summary().tables[1])
rob6 = sm.Logit(dados6["lento"], X6).fit(disp=False, cov_type="cluster", cov_kwds={"groups": grupos})
print("\nLogística com erro-padrão por órgão:\n", rob6.summary().tables[1])

# 9.3 Gráfico de resíduos do modelo múltiplo
plt.figure(figsize=(7, 5))
plt.scatter(modelo5.fittedvalues, modelo5.resid, alpha=0.6)
plt.axhline(0, color="red")
plt.title("Resíduos x valores ajustados (regressão múltipla)")
plt.xlabel("Tempo mediano previsto (dias)")
plt.ylabel("Resíduo")
finalizar_grafico("item05_residuos")

# 9.4 Validação treino/teste (70/30) e curva ROC
X_tr, X_te, y_tr, y_te = train_test_split(X6, dados6["lento"], test_size=0.30,
                                          random_state=SEED, stratify=dados6["lento"])
m_tr = sm.Logit(y_tr, X_tr).fit(disp=False)
p_te = m_tr.predict(X_te)
print("\nValidação (30% de teste):\n", confusion_matrix(y_te, (p_te > 0.5).astype(int)))
print(f"Acurácia no teste = {accuracy_score(y_te, (p_te > 0.5).astype(int)):.3f} "
      f"| AUC = {roc_auc_score(y_te, p_te):.3f}")
print(f"AUC no modelo completo = {roc_auc_score(dados6['lento'], previsoes):.3f}")

fpr, tpr, _ = roc_curve(y_te, p_te)
plt.figure(figsize=(5.5, 5))
plt.plot(fpr, tpr, label=f"AUC = {roc_auc_score(y_te, p_te):.2f}")
plt.plot([0, 1], [0, 1], "--", color="gray", label="Acaso")
plt.title("Curva ROC (dados de teste)")
plt.xlabel("Taxa de falsos positivos")
plt.ylabel("Taxa de verdadeiros positivos")
plt.legend()
finalizar_grafico("item06_roc")

# 9.5 Efeito do corte de probabilidade (sensibilidade x especificidade)
linhas = []
for corte in [0.3, 0.4, 0.5, 0.6]:
    pred = (previsoes >= corte).astype(int)
    vn_, fp_, fn_, vp_ = confusion_matrix(dados6["lento"], pred).ravel()
    linhas.append({"corte": corte, "acuracia": (vn_ + vp_) / len(pred),
                   "sensibilidade": vp_ / (vp_ + fn_), "especificidade": vn_ / (vn_ + fp_)})
tab_corte = pd.DataFrame(linhas).round(3)
print("\nEfeito do corte de probabilidade:\n", tab_corte)
tab_corte.to_csv(PASTA_TAB / "corte_probabilidade.csv", index=False)

# 9.6 Tabelas para o assistente digital: por canal, por tema e cobertura da demanda
def resumo_por(coluna, base):
    return (base.groupby(coluna)["tempo_dias"]
            .agg(n="size", mediana="median", media="mean",
                 p90=lambda s: s.quantile(0.9),
                 pct_mesmo_dia=lambda s: (s == 0).mean() * 100)
            .round(1))


por_canal = resumo_por("Meio de Comunicação", df)
por_tema = resumo_por("Tema", df).sort_values("n", ascending=False).head(10)
top_serv = resumo_por("Serviço", df).sort_values("n", ascending=False).head(10)
print("\nPor canal:\n", por_canal)
print("\nPor tema (top 10 em volume):\n", por_tema)
print("\nTop 10 serviços em volume:\n", top_serv)
por_canal.to_csv(PASTA_TAB / "por_canal.csv")
por_tema.to_csv(PASTA_TAB / "por_tema.csv")
top_serv.to_csv(PASTA_TAB / "top_servicos.csv")

vol = bruta["Serviço"].value_counts()
cum = vol.cumsum() / vol.sum() * 100
print("\nCobertura acumulada de solicitações (base bruta):")
for alvo in [50, 80, 90]:
    print(f"  {alvo}% das solicitações: {(cum < alvo).sum() + 1} serviços (de {len(vol)})")

plt.figure(figsize=(7, 5))
plt.plot(range(1, len(cum) + 1), cum.values)
plt.title("Cobertura acumulada de solicitações por serviço")
plt.xlabel("Nº de serviços (ordenados por volume)")
plt.ylabel("% acumulado das solicitações")
plt.grid(alpha=0.3)
finalizar_grafico("cobertura_servicos")

print("\nConcluído. Figuras em figuras/ e tabelas em tabelas/.")
