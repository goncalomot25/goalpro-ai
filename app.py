import streamlit as st
import cv2
import tempfile
import os
import json
import base64
import numpy as np
import pandas as pd
from openai import OpenAI

st.set_page_config(page_title="GoalPro AI Pro", page_icon="⚽", layout="wide")

# ---------------------- CSS ----------------------
st.markdown(
    """
    <style>
    .main {background: #070b12; color: #f7f7f7;}
    .stApp {background: linear-gradient(180deg, #070b12 0%, #0d1422 100%);}
    h1, h2, h3 {color: #ffffff;}
    .metric-card {background:#111827; border:1px solid #263244; border-radius:18px; padding:18px;}
    .small-muted {color:#9ca3af; font-size: 0.9rem;}
    .success-box {background:#062e1f; border:1px solid #13a36e; color:#d1fae5; padding:12px; border-radius:12px;}
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("⚽ GoalPro AI — Análise Profissional de Golos Sofridos")
st.caption("Vídeo longo → deteção/ajuste de golos → clips individuais → IA tática → dashboard/exportação")

# ---------------------- Sidebar ----------------------
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
    expected_goals = st.number_input("Nº real de golos no vídeo", min_value=1, max_value=80, value=16)
    split_mode = st.selectbox(
        "Modo de separação",
        ["Híbrido: cortes + nº de golos", "Dividir exatamente pelo nº de golos", "Só cortes de imagem"],
        index=0,
    )
    pre_margin = st.slider("Margem antes do lance", 0.0, 6.0, 1.5, 0.5)
    post_margin = st.slider("Margem depois do lance", 0.0, 8.0, 2.0, 0.5)
    frames_ai = st.slider("Frames enviados à IA por golo", 4, 18, 10)

# ---------------------- Helpers ----------------------
def seconds_to_mmss(sec):
    sec = max(0, float(sec))
    m = int(sec // 60)
    s = int(sec % 60)
    return "%02d:%02d" % (m, s)


def save_uploaded_file(uploaded_file):
    suffix = os.path.splitext(uploaded_file.name)[1] or ".mp4"
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    tmp.write(uploaded_file.getbuffer())
    tmp.close()
    return tmp.name


def get_video_info(path):
    cap = cv2.VideoCapture(path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    if not fps or fps <= 1:
        fps = 25.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / fps if fps else 0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    return fps, total_frames, duration, width, height


def scene_change_scores(path):
    cap = cv2.VideoCapture(path)
    fps, total, duration, width, height = get_video_info(path)
    step = max(1, int(fps * 0.75))
    scores = []
    prev_hist = None
    idx = 0
    while idx < total:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if not ret:
            break
        small = cv2.resize(frame, (320, 180))
        hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)
        hist = cv2.calcHist([hsv], [0, 1], None, [32, 32], [0, 180, 0, 256])
        cv2.normalize(hist, hist)
        if prev_hist is not None:
            corr = cv2.compareHist(prev_hist, hist, cv2.HISTCMP_CORREL)
            change = float(1.0 - corr)
            scores.append((idx / fps, change))
        prev_hist = hist
        idx += step
    cap.release()
    return scores


def pick_boundaries_by_expected(scores, duration, expected):
    if expected <= 1:
        return []
    min_gap = max(8.0, duration / (expected * 2.2))
    ranked = sorted(scores, key=lambda x: x[1], reverse=True)
    chosen = []
    for t, score in ranked:
        if t < 3 or t > duration - 3:
            continue
        if all(abs(t - c) >= min_gap for c in chosen):
            chosen.append(t)
        if len(chosen) >= expected - 1:
            break
    return sorted(chosen)


def pick_boundaries_threshold(scores, duration):
    if not scores:
        return []
    vals = np.array([s for _, s in scores])
    threshold = float(np.mean(vals) + 1.35 * np.std(vals))
    raw = [t for t, s in scores if s >= threshold and 3 < t < duration - 3]
    filtered = []
    for t in raw:
        if not filtered or t - filtered[-1] > 8:
            filtered.append(t)
    return filtered


def create_segments(path, expected, mode):
    fps, total, duration, width, height = get_video_info(path)
    if duration <= 0:
        return []

    if mode == "Dividir exatamente pelo nº de golos":
        boundaries = [duration * i / expected for i in range(1, expected)]
    else:
        scores = scene_change_scores(path)
        if mode == "Híbrido: cortes + nº de golos":
            boundaries = pick_boundaries_by_expected(scores, duration, expected)
            # fallback: if detection is too far from expected, divide equally
            if len(boundaries) < max(1, expected - 4):
                boundaries = [duration * i / expected for i in range(1, expected)]
        else:
            boundaries = pick_boundaries_threshold(scores, duration)

    points = [0.0] + sorted(boundaries) + [duration]
    segments = []
    for i in range(len(points) - 1):
        start = float(points[i])
        end = float(points[i + 1])
        if end - start >= 2:
            segments.append({
                "golo": len(segments) + 1,
                "inicio_s": round(start, 2),
                "fim_s": round(end, 2),
                "inicio": seconds_to_mmss(start),
                "fim": seconds_to_mmss(end),
                "duracao_s": round(end - start, 2),
            })
    return segments
    def cut_clip(path, start_s, end_s, out_path):
        import os

        start_s = max(0, start_s)
        duration = end_s - start_s

        cmd = f'ffmpeg -y -ss {start_s} -i "{path}" -t {duration} -c:v libx264 -c:a aac "{out_path}"'
        os.system(cmd)

        return out_path


def extract_frames_b64(path, n):
    cap = cv2.VideoCapture(path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frames = []
    if total <= 0:
        cap.release()
        return frames
    idxs = np.linspace(0, total - 1, n).astype(int)
    for idx in idxs:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ret, frame = cap.read()
        if ret:
            frame = cv2.resize(frame, (960, 540))
            ok, buf = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 82])
            if ok:
                frames.append("data:image/jpeg;base64," + base64.b64encode(buf).decode("utf-8"))
    cap.release()
    return frames


def analyze_clip(clip_path, clip_number, api_key, model, team_color, opponent, competition):
    client = OpenAI(api_key=api_key)
    images = extract_frames_b64(clip_path, frames_ai)
    prompt = """
És um analista profissional de futebol de alto rendimento. Analisa este clip de um golo sofrido.

Equipa analisada joga de cor: {team_color}
Adversário: {opponent}
Competição/Jogo: {competition}
Número do golo: {clip_number}

Devolve APENAS JSON válido, sem markdown, com estes campos:
{{
  "golo": {clip_number},
  "tipo_golo": "ataque posicional | ataque rápido | contra-ataque | penálti | canto direto | canto curto | livre direto | livre cruzamento | indefinido",
  "finalizacao": "cabeça | pé direito | pé esquerdo | outro | indefinido",
  "remate": "remate direto | segunda bola/recarga | indefinido",
  "passe_anterior": "profundidade | apoiado | cruzamento | bola parada | sem passe | indefinido",
  "jogadores_tua_equipa_area": 0,
  "adversarios_area": 0,
  "cor_tua_equipa_confirmada": "",
  "zona_finalizacao": "pequena área | zona central | segundo poste | entrada da área | outro | indefinido",
  "resumo_tatico": "texto curto e profissional",
  "vulnerabilidade_defensiva": "texto curto sobre o erro/padrão defensivo",
  "nivel_confianca": 0,
  "avisos": ["dúvidas relevantes"]
}}

Critérios obrigatórios:
- Ataque posicional: adversário organizado no meio-campo ofensivo, circulação, amplitude/profundidade.
- Ataque rápido: transição ofensiva veloz, vários atacantes, normalmente mais de 5.
- Contra-ataque: transição rápida com poucos jogadores, explorando defesa desorganizada.
- Canto direto: canto cruzado diretamente para a área.
- Canto curto: canto jogado curto antes da ação final.
- Livre direto: remate direto à baliza.
- Livre cruzamento: livre cruzado para a área.
- Se não for possível ver com segurança, usa "indefinido" e baixa a confiança.
""".format(team_color=team_color, opponent=opponent, competition=competition, clip_number=clip_number)

    content = [{"type": "input_text", "text": prompt}]
    for img in images:
        content.append({"type": "input_image", "image_url": img})

    try:
        resp = client.responses.create(
            model=model,
            input=[{"role": "user", "content": content}],
        )
        txt = resp.output_text.strip().replace("```json", "").replace("```", "").strip()
        return json.loads(txt)
    except Exception as e:
        return {
            "golo": clip_number,
            "tipo_golo": "erro",
            "finalizacao": "indefinido",
            "remate": "indefinido",
            "passe_anterior": "indefinido",
            "jogadores_tua_equipa_area": None,
            "adversarios_area": None,
            "cor_tua_equipa_confirmada": team_color,
            "zona_finalizacao": "indefinido",
            "resumo_tatico": "Erro na análise IA: " + str(e),
            "vulnerabilidade_defensiva": "",
            "nivel_confianca": 0,
            "avisos": ["Erro técnico na análise IA"],
        }

# ---------------------- State ----------------------
for key, default in [("segments", []), ("clips", []), ("results", []), ("video_path", None)]:
    if key not in st.session_state:
        st.session_state[key] = default

uploaded = st.file_uploader("🎥 Carrega o vídeo com todos os golos", type=["mp4", "mov", "avi", "mkv"])

if uploaded:
    video_path = save_uploaded_file(uploaded)
    st.session_state.video_path = video_path
    fps, total, duration, width, height = get_video_info(video_path)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Duração", seconds_to_mmss(duration))
    c2.metric("FPS", round(fps, 1))
    c3.metric("Resolução", "%sx%s" % (width, height))
    c4.metric("Golos esperados", expected_goals)

    st.video(video_path)

    st.header("1️⃣ Separar golos")
    if st.button("Criar lista de golos", type="primary"):
        st.session_state.segments = create_segments(video_path, expected_goals, split_mode)
        st.session_state.clips = []
        st.session_state.results = []
        st.success("Lista criada: %s golos/lances" % len(st.session_state.segments))

if st.session_state.segments:
    st.header("2️⃣ Rever/ajustar tempos")
    st.caption("Podes corrigir início/fim. Isto é o controlo profissional para garantir 1 golo = 1 clip.")
    df_segments = pd.DataFrame(st.session_state.segments)
    edited = st.data_editor(df_segments, num_rows="dynamic", use_container_width=True)
    st.session_state.segments = edited.to_dict("records")

    if st.button("Criar clips individuais"):
        if not st.session_state.video_path:
            st.error("Carrega primeiro o vídeo.")
        else:
            clip_dir = tempfile.mkdtemp(prefix="goalpro_clips_")
            fps, total, duration, width, height = get_video_info(st.session_state.video_path)
            clips = []
            with st.spinner("A criar clips..."):
                for row in st.session_state.segments:
                    golo = int(row["golo"])
                    start = max(0.0, float(row["inicio_s"]) - pre_margin)
                    end = min(duration, float(row["fim_s"]) + post_margin)
                    out = os.path.join(clip_dir, "golo_%02d.mp4" % golo)
                    cut_clip(st.session_state.video_path, start, end, out)
                    clips.append({"golo": golo, "clip": out, "inicio_s": start, "fim_s": end})
            st.session_state.clips = clips
            st.success("%s clips criados" % len(clips))

if st.session_state.clips:
    st.header("3️⃣ Biblioteca de clips")
    cols = st.columns(2)
    for i, clip in enumerate(st.session_state.clips):
        with cols[i % 2]:
            st.subheader("Golo %s" % clip["golo"])
            st.caption("%s - %s" % (seconds_to_mmss(clip["inicio_s"]), seconds_to_mmss(clip["fim_s"])))
            st.video(clip["clip"])

    st.header("4️⃣ Analisar com IA")
    if st.button("Analisar todos os clips com IA", type="primary"):
        if not api_key:
            st.error("Mete a tua OpenAI API Key na barra lateral ou nos Secrets do Streamlit.")
        else:
            results = []
            prog = st.progress(0)
            for i, clip in enumerate(st.session_state.clips):
                with st.spinner("A analisar golo %s..." % clip["golo"]):
                    r = analyze_clip(clip["clip"], clip["golo"], api_key, model, team_color, opponent, competition)
                    r["clip"] = clip["clip"]
                    r["inicio_s"] = clip["inicio_s"]
                    r["fim_s"] = clip["fim_s"]
                    results.append(r)
                prog.progress((i + 1) / len(st.session_state.clips))
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

    st.subheader("Tabela de análise")
    st.dataframe(df.drop(columns=["clip"], errors="ignore"), use_container_width=True)

    csv = df.drop(columns=["clip"], errors="ignore").to_csv(index=False).encode("utf-8")
    st.download_button("⬇️ Download CSV", csv, "analise_golos_goalpro.csv", "text/csv")
    st.download_button("⬇️ Download JSON", json.dumps(st.session_state.results, ensure_ascii=False, indent=2), "analise_golos_goalpro.json", "application/json")

    st.subheader("Relatório por golo")
    for r in st.session_state.results:
        with st.expander("Golo %s — %s" % (r.get("golo"), r.get("tipo_golo"))):
            if r.get("clip"):
                st.video(r.get("clip"))
            st.json(r)
else:
    st.info("Fluxo: carregar vídeo → criar lista de golos → ajustar tempos → criar clips → analisar com IA → exportar.")
