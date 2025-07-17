# DAFdiscovery App

**Análise Integrada de NMR, MS e Bioatividade via STOCSY**

Aplicação interativa desenvolvida em **Streamlit** para análise multivariada de dados de **RMN (NMR)**, **Espectrometria de Massas (MS)** e **Bioatividade**, utilizando o método **STOCSY (Statistical Total Correlation Spectroscopy)**.

## Funcionalidades

- Upload de dados e metadados (`.csv`)
- Integração automática dos dados: `NMR + MS + BioActivity`
- Execução de STOCSY com diferentes modelos de correlação (linear, sigmoidal, etc.)
- Seleção de drivers automáticos (bioatividade) ou manuais (ppm ou índice MS)
- Visualizações interativas com Plotly e mpld3
- Exportação de resultados em PDF, HTML e CSV

## Instalação

Clone o repositório e instale as dependências:

```bash
git clone https://github.com/SEU_USUARIO/DAFdiscovery-App.git
cd DAFdiscovery-App
pip install -r requirements.txt
```

## Como executar
```bash
streamlit run app.py
```
