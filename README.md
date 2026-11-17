# CitizenTech_SP156
Análise estatística em Python de mais de 157 mil solicitações reais da Central SP156 de São Paulo, para descobrir o que faz um serviço público demorar mais. Os resultados orientam um assistente digital que ajuda o cidadão a encontrar serviços e a saber o que esperar de cada um. 

CitizenTech | Tempo de atendimento da Central SP156

Análise estatística em Python de 157.891 solicitações reais da Central SP156 da Prefeitura de São Paulo (julho/2026), feita para o desafio CitizenTech (FIAP). O objetivo é descobrir o que faz um serviço público demorar mais e como isso deve orientar um assistente digital que ajude o cidadão a encontrar informações sobre serviços.

Principais resultados
O tempo de atendimento é muito assimétrico: mediana de 3 dias, média de 9,25 e 27% das solicitações resolvidas no mesmo dia.
O fator mais associado à demora de um serviço é a fila do órgão responsável (% de solicitações não finalizadas): r = 0,55, R² = 0,31. Cada ponto percentual a mais está associado a cerca de 0,36 dia a mais. O resultado se mantém com erros-padrão agrupados por órgão.
Um modelo de regressão logística classifica serviços como lentos ou não lentos com 70,9% de acurácia (referência de 50,6%) e AUC de 0,76; na validação, 72,0%.
A demanda é concentrada: 58 de 601 serviços respondem por 80% das solicitações, o que sustenta um assistente com escopo inicial enxuto.
O que foi feito
Etapa	Técnica
Estatística descritiva	medidas-resumo, tabela de frequências (Sturges), boxplot, normalidade (Shapiro-Wilk, QQ-plot)
Probabilidade	distribuição Normal e classificação de eventos
Teste de hipótese	teste Z para a média, com amostras de semente fixa
Análise bivariada	dispersão, covariância, correlação de Pearson, regressão linear simples, R² e teste F
Análise multivariada	regressão linear múltipla, VIF, erros-padrão agrupados por órgão
Classificação	regressão logística, matriz de confusão, acurácia, ROC/AUC, validação treino/teste
Insights	cobertura da demanda, tempos por canal e por tema, recomendações para o assistente
Estrutura
citizentech-sp156/
├── dados/      # base SP156 (CSV) e base agregada por serviço
├── src/        # citizentech_sp156.py (pipeline completo)
├── figuras/    # gráficos gerados pelo script
├── tabelas/    # tabelas de apoio (canal, tema, cortes de probabilidade)
└── relatorio/  # relatório estatístico (PDF)
Como executar
bash
pip install pandas numpy scipy statsmodels scikit-learn seaborn matplotlib
python src/citizentech_sp156.py

O script usa semente fixa (42), então os resultados são reproduzíveis. Para só salvar os gráficos, sem abrir janelas, defina MOSTRAR_GRAFICOS = False no topo do arquivo.

Dados

Dados Abertos da Prefeitura de São Paulo (SP156), solicitações abertas entre 01/07/2026 e 31/07/2026. São dados reais. A análise não usa campos de localização (logradouro, CEP, coordenadas), em linha com o princípio de não tratar dados pessoais desnecessários.

Limitações
O tempo medido é o do registro de finalização, que não equivale necessariamente à resolução do problema.
Solicitações ainda abertas ficam fora do cálculo, o que tende a subestimar os tempos.
Um único mês de dados, sem sazonalidade.
A associação com a fila do órgão é um sinal de alerta, não uma relação de causa.
English summary

Statistical analysis in Python of 157,891 real service requests from São Paulo's SP156 citizen-service center (July 2026), built for the CitizenTech challenge at FIAP. The goal is to find what makes a public service slower and how a citizen-facing digital assistant should use that. Main finding: the backlog of the responsible agency (share of unfinished requests) is the strongest predictor of service delay (r = 0.55, R² = 0.31, robust to clustering by agency); a logistic model classifies slow vs. non-slow services with 70.9% accuracy (50.6% baseline, AUC 0.76). Demand is concentrated: 58 of 601 services cover 80% of requests. Run python src/citizentech_sp156.py to reproduce everything (fixed seed).
