# GoalPro AI Online

App online para análise profissional de golos sofridos.

## Ficheiros
- `app.py` — aplicação Streamlit
- `requirements.txt` — dependências

## Deploy recomendado: Streamlit Community Cloud
1. Cria um repositório no GitHub.
2. Faz upload destes ficheiros.
3. Vai a Streamlit Community Cloud e cria uma app a partir do repositório.
4. Em Advanced settings > Secrets, adiciona:

```toml
OPENAI_API_KEY="a_tua_chave"
```

5. Entry point: `app.py`

## Uso
1. Carrega o vídeo.
2. Indica o nº real de golos.
3. Cria a lista de golos.
4. Ajusta os tempos se necessário.
5. Cria clips individuais.
6. Analisa todos com IA.
7. Exporta CSV/JSON.
