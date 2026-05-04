import base64
import json
import tempfile
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import streamlit as st
from openai import OpenAI

st.set_page_config(page_title="GoalPro AI — Hashtag", page_icon="⚽", layout="wide")

st.markdown(
    """
    <style>
    .stApp {background: linear-gradient(180deg, #07101f 0%, #0b1220 100%); color: #f8fafc;}
    h1,h2,h3 {color: #ffffff;}
    .small {color:#9ca3af; font-size:0.92rem;}
    .okbox {background:#063b26; border:1px solid #15803d; padding:12px; border-radius:12px; color:#d1fae5;}
    .warnbox {background:#3b2506; border:1px solid #d97706; padding:12px; border-radius:12px; color:#ffedd5;}
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("⚽ GoalPro AI — análise profissional de golos sofridos")
st.caption("Versão estável: separa golos pela mudança do marcador # no canto inferior esquerdo")

# -------------------- Sidebar --------------------
with st.sidebar:
    st.header("⚙️ Configuração")
    default_key = ""
    try:
        default_key = st.secrets.get("OPENAI_API_KEY", "")
    except Exception:
        default_key = ""

    api_key = st.text_input("OpenAI API Key", value=default_key, type="password")
    model = st.selectbox("Modelo IA", ["gpt-4.1", "gpt-4.1-mini"], index=0)
    team_color = st.text_input("Cor da tua equipa", placeholder="ex: branco, azul, vermelho")
    opponent = st.text_input("Adversário", placeholder="ex: Benfica")
    competition = st.text_input("Jogo/competição", placeholder="ex: Liga Sub-19, Jornada 12")

    st.divider()
    st.subheader("Separação pelo #")
    sample_every = st.slider("Amostragem do vídeo (segundos)", 0.5, 3.0, 1.0, 0.5)
    sensitivity = st.slider("Sensibilidade à mudança do #", 5, 60, 22)
    min_gap = st.slider("Intervalo mínimo entre golos (segundos)", 8, 60, 15)
    before_margin = st.slider("Começar antes do # mudar (segundos)", 0.0, 10.0, 3.0, 0.5)
    after_margin = st.slider("Terminar depois do próximo # mudar (segundos)", 0.0, 10.0, 2.0, 0.5)
    frames_ai = st.slider("Frames enviados à IA por golo", 4, 16, 10)

# -------------------- Helpers --------------------
def seconds_to_mmss(sec: float) -> str:
    sec = max(0, float(sec))
    m = int(sec // 60)
    s = int(sec % 60)
    return f"{m:02d}:{s:02d}"


def save_upload(uploaded_file) -> str:
    suffix = Path(uploaded_file.name).suffix or ".mp4"
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    tmp.write(uploaded_file.read())
    tmp.flush()
    return tmp.name


def get_video_info(path: str):
    cap = cv2.VideoCapture(path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    duration = frames / fps if fps else 0
    cap.release()
    return fps, frames, duration, width, height


def bottom_left_roi(frame):
    """Crop da zona onde normalmente aparece o # do golo."""
    h, w = frame.shape[:2]
    # canto inferior esquerdo: 0-38% largura, 70-100% altura
    x1, x2 = 0, int(w * 0.38)
    y1, y2 = int(h * 0.70), h
    return frame[y1:y2, x1:x2]


def crop_score(roi):
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, (320, 120))
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    return gray


def detect_hashtag_changes(path: str, sample_sec: float, sensitivity: int, min_gap_sec: int):
    """Detecta mudanças visuais no canto inferior esquerdo, onde o marcador # muda.
    Não faz OCR; detecta a mudança gráfica do bloco #6 -> #7, etc.
    """
    cap = cv2.VideoCapture(path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    duration = total / fps if fps else 0
    step = max(1, int(fps * sample_sec))

    prev = None
    candidates = []
    scores = []

    frame_idx = 0
    while frame_idx < total:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ok, frame = cap.read()
        if not ok:
            break
        roi = crop_score(bottom_left_roi(frame))
        if prev is not None:
            diff = cv2.absdiff(prev, roi)
            score = float(np.mean(diff))
            t = frame_idx / fps
            scores.append((t, score))
            if score >= sensitivity:
                candidates.append(t)
        prev = roi
        frame_idx += step
    cap.release()

    # agrupa mudanças próximas
    changes = []
    for t in candidates:
        if not changes or (t - changes[-1]) >= min_gap_sec:
            changes.append(t)

    return changes, scores, duration


def segments_from_changes(changes, duration, before=3.0, after=2.0):
    if not changes:
        return []
    # O primeiro golo começa no início do vídeo; mudanças são passagem #6 -> #7 etc.
    boundaries = [0.0] + list(changes) + [duration]
    segments = []
    for i in range(len(boundaries) - 1):
        start = max(0.0, boundaries[i] - (before if i > 0 else 0.0))
        end = min(duration, boundaries[i + 1] + after)
        if end - start >= 3:
            segments.append({
                "golo": i + 1,
                "inicio_s": round(start, 2),
                "fim_s": round(end, 2),
                "inicio": seconds_to_mmss(start),
                "fim": seconds_to_mmss(end),
                "duracao_s": round(end - start, 2),
            })
    return segments


def extract_frames_b64(path: str, start_s: float, end_s: float, n: int):
    cap = cv2.VideoCapture(path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    start_frame = int(max(0, start_s) * fps)
    end_frame = int(min(end_s * fps, max(0, total - 1)))
    if end_frame <= start_frame:
        end_frame = min(total - 1, start_frame + int(5 * fps))
    idxs = np.linspace(start_frame, end_frame, n).astype(int)
    images = []
    for idx in idxs:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ok, frame = cap.read()
        if ok:
            frame = cv2.resize(frame, (960, 540))
            ok2, buf = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 82])
            if ok2:
                images.append("data:image/jpeg;base64," + base64.b64encode(buf).decode("utf-8"))
    cap.release()
    return images


def analyze_goal(path, seg, api_key, model, team_color, opponent, competition, frames_ai):
    client = OpenAI(api_key=api_key)
    imgs = extract_frames_b64(path, seg["inicio_s"], seg["fim_s"], frames_ai)
    prompt = f"""
És um analista profissional de futebol de alto rendimento.
Analisa este intervalo de vídeo de um golo sofrido e devolve APENAS JSON válido, sem markdown.

Contexto:
- Equipa analisada joga de cor: {team_color}
- Adversário: {opponent}
- Competição/Jogo: {competition}
- Golo nº: {seg['golo']}
- Intervalo no vídeo: {seg['inicio']} a {seg['fim']}

Campos obrigatórios:
{{
  "golo": {seg['golo']},
  "tipo_golo": "ataque posicional | ataque rápido | contra-ataque | penálti | canto direto | canto curto | livre direto | livre cruzamento | indefinido",
  "finalizacao": "cabeça | pé direito | pé esquerdo | outro | indefinido",
  "remate": "remate direto | segunda bola/recarga | indefinido",
  "passe_anterior": "profundidade | apoiado | cruzamento | bola parada | sem passe | indefinido",
  "jogadores_tua_equipa_area": 0,
  "adversarios_area": 0,
  "cor_tua_equipa_confirmada": "texto curto",
  "zona_finalizacao": "pequena área | zona central | segundo poste | entrada da área | outro | indefinido",
  "resumo_tatico": "texto curto e profissional",
  "vulnerabilidade_defensiva": "texto curto sobre o erro/padrão defensivo",
  "nivel_confianca": 0,
  "avisos": ["dúvidas relevantes"]
}}

Critérios:
- Ataque posicional: adversário organizado no meio-campo ofensivo, circulação, amplitude/profundidade.
- Ataque rápido: transição ofensiva veloz, vários atacantes, normalmente mais de 5.
- Contra-ataque: transição rápida com poucos jogadores, explorando defesa desorganizada.
- Canto direto: canto cruzado diretamente para a área.
- Canto curto: canto jogado curto antes da ação final.
- Livre direto: remate direto à baliza.
- Livre cruzamento: livre cruzado para a área.
- Se não for possível ver com segurança, usa "indefinido" e baixa a confiança.
"""
    content = [{"type": "input_text", "text": prompt}]
    for img in imgs:
        content.append({"type": "input_image", "image_url": img})
    try:
        resp = client.responses.create(model=model, input=[{"role": "user", "content": content}])
        txt = resp.output_text.strip().replace("```json", "").replace("```", "").strip()
        return json.loads(txt)
    except Exception as e:
        return {
            "golo": seg["golo"], "tipo_golo": "erro", "finalizacao": "indefinido", "remate": "indefinido",
            "passe_anterior": "indefinido", "jogadores_tua_equipa_area": None, "adversarios_area": None,
            "cor_tua_equipa_confirmada": team_color, "zona_finalizacao": "indefinido",
            "resumo_tatico": "Erro na análise IA: " + str(e), "vulnerabilidade_defensiva": "",
            "nivel_confianca": 0, "avisos": ["erro técnico"]
        }

# -------------------- State --------------------
for k, v in {
    "video_path": None,
    "segments": [],
    "results": [],
    "duration": 0.0,
    "scores": [],
    "changes": [],
}.items():
    if k not in st.session_state:
        st.session_state[k] = v

uploaded = st.file_uploader("🎥 Carrega o vídeo com todos os golos", type=["mp4", "mov", "avi", "mkv"])

if uploaded:
    if st.session_state.video_path is None or st.session_state.get("uploaded_name") != uploaded.name:
        st.session_state.video_path = save_upload(uploaded)
        st.session_state.uploaded_name = uploaded.name
        st.session_state.segments = []
        st.session_state.results = []
    fps, total, duration, width, height = get_video_info(st.session_state.video_path)
    st.session_state.duration = duration

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Duração", seconds_to_mmss(duration))
    c2.metric("FPS", round(fps, 1))
    c3.metric("Resolução", f"{width}x{height}")
    c4.metric("Frames", total)

    st.video(st.session_state.video_path)

    st.header("1️⃣ Separar golos pelo marcador #")
    st.markdown("<div class='small'>A app observa o canto inferior esquerdo. Quando o # muda, cria um novo golo.</div>", unsafe_allow_html=True)

    if st.button("Detetar mudanças do #", type="primary"):
        with st.spinner("A detetar mudanças do marcador # no canto inferior esquerdo..."):
            changes, scores, dur = detect_hashtag_changes(st.session_state.video_path, sample_every, sensitivity, min_gap)
            segments = segments_from_changes(changes, duration, before_margin, after_margin)
            st.session_state.changes = changes
            st.session_state.scores = scores
            st.session_state.segments = segments
            st.session_state.results = []
        st.success(f"{len(st.session_state.segments)} golos/lances criados a partir do marcador #")

    if st.session_state.segments:
        st.header("2️⃣ Rever/ajustar tempos")
        st.caption("Podes corrigir início/fim manualmente. Isto é o controlo profissional para garantir 1 golo = 1 intervalo.")
        df_segments = pd.DataFrame(st.session_state.segments)
        edited = st.data_editor(df_segments, num_rows="dynamic", use_container_width=True)
        # recalcular mm:ss caso editem segundos
        segs = edited.to_dict("records")
        for s in segs:
            s["inicio"] = seconds_to_mmss(float(s["inicio_s"]))
            s["fim"] = seconds_to_mmss(float(s["fim_s"]))
            s["duracao_s"] = round(float(s["fim_s"]) - float(s["inicio_s"]), 2)
        st.session_state.segments = segs

        st.header("3️⃣ Ver cada golo")
        cols = st.columns(2)
        for i, seg in enumerate(st.session_state.segments):
            with cols[i % 2]:
                st.subheader(f"Golo {int(seg['golo'])}")
                st.caption(f"{seg['inicio']} – {seg['fim']}")
                if st.button(f"▶️ Ver golo {int(seg['golo'])}", key=f"see_{i}"):
                    st.video(st.session_state.video_path, start_time=int(float(seg["inicio_s"])))

        st.header("4️⃣ Analisar com IA")
        if st.button("Analisar todos os golos com IA", type="primary"):
            if not api_key:
                st.error("Mete a tua OpenAI API Key na barra lateral ou nos Secrets do Streamlit.")
            else:
                results = []
                prog = st.progress(0)
                for i, seg in enumerate(st.session_state.segments):
                    with st.spinner(f"A analisar golo {seg['golo']}..."):
                        r = analyze_goal(st.session_state.video_path, seg, api_key, model, team_color, opponent, competition, frames_ai)
                        r["inicio"] = seg["inicio"]
                        r["fim"] = seg["fim"]
                        r["inicio_s"] = seg["inicio_s"]
                        r["fim_s"] = seg["fim_s"]
                        results.append(r)
                    prog.progress((i + 1) / len(st.session_state.segments))
                st.session_state.results = results
                st.success("Análise concluída")

if st.session_state.results:
    st.header("📊 Dashboard final")
    df = pd.DataFrame(st.session_state.results)
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Golos analisados", len(df))
    col2.metric("Bola parada", int(df["tipo_golo"].astype(str).str.contains("canto|livre|penálti", case=False, na=False).sum()))
    col3.metric("Transições", int(df["tipo_golo"].astype(str).str.contains("contra|rápido", case=False, na=False).sum()))
    col4.metric("Confiança média", round(pd.to_numeric(df["nivel_confianca"], errors="coerce").mean(), 1))

    st.subheader("Tabela final")
    st.dataframe(df, use_container_width=True)

    st.download_button("⬇️ Download CSV", df.to_csv(index=False).encode("utf-8"), "analise_golos_goalpro.csv", "text/csv")
    st.download_button("⬇️ Download JSON", json.dumps(st.session_state.results, ensure_ascii=False, indent=2), "analise_golos_goalpro.json", "application/json")

    st.subheader("Relatório por golo")
    for r in st.session_state.results:
        with st.expander(f"Golo {r.get('golo')} — {r.get('tipo_golo')}"):
            st.write(f"**Tempo:** {r.get('inicio')} – {r.get('fim')}")
            if st.button(f"▶️ Ver golo {r.get('golo')}", key=f"report_{r.get('golo')}"):
                st.video(st.session_state.video_path, start_time=int(float(r.get("inicio_s", 0))))
            st.json(r)
else:
    st.info("Fluxo: carregar vídeo → detectar mudanças do # → ajustar tempos → ver golos → analisar com IA → exportar.")
