# GoalPro AI — Detecção pelo marcador #

Esta versão separa os golos usando a mudança do marcador `#` no canto inferior esquerdo do vídeo.
Não corta clips, não usa ffmpeg e evita erros no Streamlit Cloud.

Fluxo:
1. Upload do vídeo.
2. Escolher modo: detectar pelo # no canto inferior esquerdo.
3. Rever/ajustar tempos.
4. Ver cada golo no vídeo a partir do tempo certo.
5. Analisar com IA.
6. Exportar CSV/JSON.
