# Projeto Final Algotrading  
**Retornos, Order Flow e Anúncios Macroeconômicos: Uma Análise Estrutural em Alta Frequência**  
**Autor:** Diego Azenha • Insper • 2025

---

## 📌 Visão Geral

Este repositório contém o código, os dados processados e os resultados do projeto que replica e adapta o arcabouço de **Takahashi (2025)** ao mercado futuro de S&P 500 (E-mini).  
O objetivo é estimar a relação contemporânea e dinâmica entre **retornos** e **Order Flow Imbalance (OFI)** usando um **SVAR bivariado identificado via heterocedasticidade (ITH)** sobre dados BBO reamostrados a 1 segundo.

Principais entregáveis:
- Pipeline de preparação de dados (BBO → BBO-1s)  
- Cálculo de OFI, mid-quote e retornos  
- Segmentação intradiária em janelas de 15 minutos  
- Estimação do SVAR–ITH por janela  
- Geração de estatísticas, regressões e IRFs  
- Gráficos e tabelas prontos para relatório/Overleaf

---

## 📂 Estrutura do repositório

```text
analysis_outputs/
    figures/             # Gráficos finais (PNG) usados no relatório
    tables/              # Tabelas finais (CSV e LaTeX)
    clean_data/          # Dados intermediários pós-limpeza
economic_releases/
    calendar_scan.py     # utilitário para varrer/formatar calendário de anúncios
    calendar.txt
    macro_announcements_*.csv
models/
    # arquivos de resultados do SVAR/IRFs (npz/parquet)
scripts/
    01_windowing.py      # cria janelas intradiárias a partir de BBO raw
    02_descriptive_stats.py
    03_estimate_svar.py  # núcleo: estima SVAR via ITH por janela
    04_aggregate_and_plot.py
windows_parquet/
    descriptives/        # parquet com descritivas 1s por janela
    windows_all_days.parquet
README.md
```


---

## 🛠️ Como rodar (executar o pipeline)

Recomenda-se usar um ambiente virtual Python com pacotes listados no `requirements.txt` (crie se necessário). Exemplo de sequência:

```bash
# ativar venv (exemplo)
python -m venv .venv
source .venv/bin/activate       # Linux / macOS
# Windows PowerShell:
# .venv\Scripts\Activate.ps1

pip install -r requirements.txt

# 1) Preparar janelas (de raw BBO -> parquet de janelas)
python scripts/01_windowing.py --input /path/to/raw_bbo --out windows_parquet/

# 2) Estatísticas descritivas
python scripts/02_descriptive_stats.py --windows windows_parquet/descriptives.parquet --out analysis_outputs/tables/

# 3) Estimar SVAR em cada janela
python scripts/03_estimate_svar.py --windows windows_parquet/ --out models/

# 4) Agregar resultados e plotar
python scripts/04_aggregate_and_plot.py --models models/ --out analysis_outputs/figures/
```

> Observações:
> - Muitos scripts aceitam argumentos de input/output. Rode `--help` para ver opções.  
> - O pipeline assume que os dados raw BBO vêm em formato compatível (colunas: ts_event, bid_px, ask_px, bid_sz, ask_sz, ...). Ver `scripts/01_windowing.py` para o formato exigido.

---

## 📈 Saídas e arquivos importantes

```text
analysis_outputs/figures/        # PNG com figuras finais do paper (intraday, pre/post, IRFs)
analysis_outputs/tables/         # CSV / LaTeX com tabelas enumeradas (Tabela 1..3)
models/                          # resultados do SVAR (parquet/npz por janela)
windows_parquet/                 # janelas 15-min em parquet (usadas pelos scripts)
economic_releases/               # calendários e arquivos de anúncios macro
```


---

## 🔎 Reprodutibilidade e notas técnicas

- **Identificação**: o SVAR é identificado via heterocedasticidade (ITH) seguindo Takahashi (2025). O método exige variação suficiente nas variâncias entre subestados; por isso cada janela de 15 minutos é particionada em subintervalos para gerar estados com volatilidades distintas.  
- **Frequência**: usamos BBO reamostrado a 1 segundo (BBO-1s). Isso preserva a maioria dos sinais intradiários, mas comprime eventos intrassegundos — limitação discutida no relatório.  
- **Depth vs Average Size**: há uma verificação no pipeline para evitar duplicação entre variáveis Depth e Average Size — confirme nas saídas de `02_descriptive_stats.py` se os valores fazem sentido.  
- **Performance**: `03_estimate_svar.py` é paralelizado; ajuste `--nprocs` conforme CPU disponível.

---

## ✅ Check-list para submissão / reprodução

```text
- [ ] Ter raw BBO com timestamps e tamanhos corretos
- [ ] Atualizar caminhos em scripts/params
- [ ] Executar 01_windowing.py para gerar windows_parquet/
- [ ] Executar 02_descriptive_stats.py e revisar tabelas
- [ ] Executar 03_estimate_svar.py (pode demorar conforme número de janelas)
- [ ] Executar 04_aggregate_and_plot.py para gerar figuras finais
```


---

## 📚 Referências principais

- Takahashi (2025), *[título do artigo]* — original do método ITH aplicado a microestrutura.  
- Cont, Kukanov & Stoikov (2014), *Order Flow Imbalance and Price Impact* — definição e construção do OFI.

---

## 🤝 Contribuições e uso

- Código desenvolvido como projeto acadêmico. Fique à vontade para abrir issues e pull requests.  
- Se utilizar este repositório em trabalhos acadêmicos, cite o autor e as bases de dados originais.

---

## ✉️ Contato

Diego Azenha — diegoa4@al.insper.edu.br


