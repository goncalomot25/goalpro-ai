import base64
import json
import os
import tempfile
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import streamlit as st
from openai import OpenAI

st.set_page_config(page_title="GoalPro AI Online", page_icon="⚽", layout="wide")

st.markdown("""
<style>
.block-container {padding-top: 2rem;}
.metric-card {background:#111827;border:1px solid #243044;border-radius:18px;padding:18px;color:white;}
.small-muted {color:#9CA3AF;font-size:0.9rem;}
</style>
""", unsafe_allow_html=True)

st.title("⚽ GoalPro AI Online")
st.caption("Análise profissional de golos sofridos: vídeo da época → clips → IA → dashboard → exportação")

# -----------------------------
# Helpers
# -----------------------------
def get_api_key():
    try:
        return st.secrets.get("OPENAI_API_KEY", "")
    except Exception:
        return ""


def save_uploaded_file(uploaded_file):
    suffix = Path(uploaded_file.name).suffix or ".mp4"
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    tmp.write(uploaded_file.getvalue())
    tmp.close()
    return tmp.name


def video_meta(video_path):
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    duration = total / fps if fps else 0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    cap.release()
    return fps, total, duration, width, height


def format_time(seconds):
    seconds = max(0, int(seconds))
    m, s = divmod(seconds, 60)
    return f"{m:02d}:{s:02d}"


def detect_candidate_boundaries(video_path, expected_goals):
    """Detect likely boundaries between edited goal clips using histogram jumps.
    If expected_goals is given, select expected_goals-1 strongest well-spaced jumps.
    """
    fps, total, duration, _, _ = video_meta(video_path)
    if total <= 0:
        return []

    cap = cv2.VideoCapture(video_path)
    sample_every = max(1, int(fps * 0.8))
    scores = []
    prev_hist = None

    for frame_idx in range(0, total, sample_every):
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ok, frame = cap.read()
        if not ok:
            continue
        frame = cv2.resize(frame, (320, 180))
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        hist = cv2.calcHist([hsv], [0, 1], None, [32, 32], [0, 180, 0, 256])
        cv2.normalize(hist, hist)
        if prev_hist is not None:
            corr = cv2.compareHist(prev_hist, hist, cv2.HISTCMP_CORREL)
            scores.append((frame_idx / fps, max(0, 1 - corr)))
        prev_hist = hist

    cap.release()
    if not scores:
        return []

    if expected_goals and expected_goals > 1:
        # Pick strong boundaries, avoiding boundaries too close together.
        ranked = sorted(scores, key=lambda x: x[1], reverse=True)
        min_gap = max(5.0, duration / (expected_goals * 2.2))
        chosen = []
        for t, score in ranked:
            if t < 3 or t > duration - 3:
                continue
            if all(abs(t - c) >= min_gap for c in chosen):
                chosen.append(t)
            if len(chosen) >= expected_goals - 1:
                break
        return sorted(chosen)

    vals = np.array([score for _, score in scores])
    threshold = float(vals.mean() + vals.std() * 1.4)
    raw = [t for t, score in scores if score >= threshold]
    filtered = []
    for t in raw:
        if not filtered or t - filtered[-1] >= 8:
            filtered.append(t)
    return filtered


def make_segments(video_path, expected_goals, mode):
    fps, total, duration, _, _ = video_meta(video_path)
    if not duration:
        return []
    if mode == "Dividir exatamente pelo nº de golos":
        step = duration / expected_goals
        points = [i * step for i in range(expected_goals + 1)]
    else:
        boundaries = detect_candidate_boundaries(video_path, expected_goals)
        points = [0.0] + boundaries + [duration]
        # fallback: ensure expected number of segments
        if expected_goals and len(points) - 1 != expected_goals:
            step = duration / expected_goals
            points = [i * step for i in range(expected_goals + 1)]

    rows = []
    for i in range(len(points) - 1):
        start = max(0.0, float(points[i]))
        end = min(duration, float(points[i + 1]))
        if end - start >= 1:
            rows.append({
                "golo": i + 1,
                "inicio_s": round(start, 2),
                "fim_s": round(end, 2),
                "inicio": format_time(start),
                "fim": format_time(end),
                "duracao_s": round(end - start, 2),
            })
    return rows


def cut_clip(video_path, start_s, end_s, clip_name):
    fps, total, duration, width, height = video_meta(video_path)
    out_dir = Path(tempfile.gettempdir()) / "goalpro_clips"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = str(out_dir / clip_name)

    cap = cv2.VideoCapture(video_path)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(out_path, fourcc, fps, (width, height))
    start_frame = int(max(0, start_s) * fps)
    end_frame = int(min(duration, end_s) * fps)
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    current = start_frame
    while current <= end_frame:
        ok, frame = cap.read()
        if not ok:
            break
        writer.write(frame)
        current += 1
    cap.release()
    writer.release()
    return out_path


def frame_images_base64(video_path, n_frames):
    cap = cv2.VideoCapture(video_path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    if total <= 0:
        return []
    idxs = np.linspace(0, total - 1, n_frames).astype(int)
    images = []
    for idx in idxs:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ok, frame = cap.read()
        if not ok:
            continue
        frame = cv2.resize(frame, (960, 540))
        ok, buf = cv2.imencode(".jpg", frame)
        if ok:
            images.append("data:image/jpeg;base64," + base64.b64encode(buf).decode("utf-8"))
    cap.release()
    return images


def safe_json_loads(text):
    text = text.strip().replace("```json", "").replace("```", "").strip()
    start = text.find("{")
    end = text.rfind("}")
    if start >= 0 and end > start:
        text = text[start:end + 1]
    return json.loads(text)


def analyze_clip(clip_path, api_key, team_color, opponent, match_name, n_frames):
    client = OpenAI(api_key=api_key)
    images = frame_images_base64(clip_path, n_frames)

    prompt = f"""
És um analista profissional de futebol de alto rendimento. Analisa este clip de um golo sofrido.

Equipa analisada: joga de {team_color or 'cor não indicada'}.
Adversário: {opponent or 'não indicado'}.
Jogo/competição: {match_name or 'não indicado'}.

Devolve APENAS JSON válido, sem markdown, com estas chaves:
{{
  "tipo_golo": "ataque posicional | ataque rápido | contra-ataque | penálti | canto direto | canto curto | livre direto | livre cruzamento | indefinido",
  "finalizacao": "cabeça | pé direito | pé esquerdo | outro | indefinido",
  "tipo_remate": "remate direto | segunda bola/recarga | indefinido",
  "passe_anterior": "profundidade | apoiado | cruzamento | bola parada | sem passe | indefinido",
  "jogadores_tua_equipa_area": 0,
  "adversarios_area": 0,
  "cor_tua_equipa_confirmada": "string",
  "resumo_tatico": "texto curto e profissional",
  "vulnerabilidade_defensiva": "texto curto",
  "nivel_confianca": 0,
  "avisos": ["dúvidas ou limitações visuais"]
}}

Definições obrigatórias:
- Ataque posicional: equipa organizada no meio-campo adversário, circulação curta, ocupação racional, amplitude/profundidade.
- Ataque rápido: transição veloz com muitos atacantes, normalmente mais de 5, antes da defesa reorganizar.
- Contra-ataque: transição rápida com poucos jogadores, explorando defesa desorganizada e espaço vazio.
- Canto direto: canto cruzado diretamente para a área.
- Canto curto: canto jogado curto antes de nova ação.
- Livre direto: remate direto à baliza.
- Livre cruzamento: livre cruzado para a área.

Sê conservador: se não for visível, usa "indefinido" e explica em avisos.
"""
    content = [{"type": "input_text", "text": prompt}]
    for img in images:
        content.append({"type": "input_image", "image_url": img})

    response = client.responses.create(
        model="gpt-4.1",
        input=[{"role": "user", "content": content}],
    )
    return safe_json_loads(response.output_text)

# -----------------------------
# State
# -----------------------------
for key, default in {
    "video_path": None,
    "segments": [],
    "clips": [],
    "results": [],
}.items():
    if key not in st.session_state:
        st.session_state[key] = default

# -----------------------------
# Sidebar
# -----------------------------
with st.sidebar:
    st.header("⚙️ Configuração")
    secret_key = get_api_key()
    api_key = st.text_input("OpenAI API Key", type="password", value=secret_key)
    st.caption("Online: podes guardar isto em Secrets como OPENAI_API_KEY.")

    team_color = st.text_input("Cor da tua equipa", placeholder="ex: branco, azul, vermelho")
    opponent = st.text_input("Adversário", placeholder="ex: Benfica")
    match_name = st.text_input("Jogo/competição", placeholder="ex: Liga Sub-19, Jornada 12")

    expected_goals = st.number_input("Nº real de golos no vídeo", min_value=1, max_value=80, value=16)
    segment_mode = st.selectbox(
        "Modo de separação",
        ["Dividir exatamente pelo nº de golos", "Tentar detetar cortes + ajustar ao nº de golos"],
        index=0,
    )
    margin_before = st.slider("Margem antes do lance", 0.0, 8.0, 1.0, 0.5)
    margin_after = st.slider("Margem depois do lance", 0.0, 8.0, 1.5, 0.5)
    n_frames = st.slider("Frames enviados à IA por golo", 4, 18, 10)

# -----------------------------
# Main app
# -----------------------------
video = st.file_uploader("Carrega o vídeo com os golos", type=["mp4", "mov", "avi", "mkv"])

if video:
    st.session_state.video_path = save_uploaded_file(video)
    fps, total, duration, width, height = video_meta(st.session_state.video_path)

    col_a, col_b, col_c = st.columns(3)
    col_a.metric("Duração", format_time(duration))
    col_b.metric("FPS", round(fps, 1))
    col_c.metric("Resolução", f"{width}x{height}")

    st.video(st.session_state.video_path)

    st.divider()
    st.subheader("1️⃣ Separar golos")
    if st.button("Criar lista de golos", type="primary"):
        st.session_state.segments = make_segments(st.session_state.video_path, int(expected_goals), segment_mode)
        st.session_state.clips = []
        st.session_state.results = []
        st.success(f"Foram criados {len(st.session_state.segments)} segmentos. Podes ajustar os tempos antes de criar clips.")

if st.session_state.segments:
    st.subheader("2️⃣ Rever/ajustar tempos")
    st.caption("Podes corrigir início/fim se algum golo ficou mal dividido. Isto é o controlo profissional para garantir 1 golo = 1 clip.")
    df_segments = pd.DataFrame(st.session_state.segments)
    edited = st.data_editor(
        df_segments,
        use_container_width=True,
        num_rows="dynamic",
        column_config={
            "inicio_s": st.column_config.NumberColumn("Início (s)", step=0.5),
            "fim_s": st.column_config.NumberColumn("Fim (s)", step=0.5),
        },
    )
    st.session_state.segments = edited.to_dict("records")

    if st.button("Criar clips individuais"):
        st.session_state.clips = []
        _, _, duration, _, _ = video_meta(st.session_state.video_path)
        for row in st.session_state.segments:
            golo = int(row["golo"])
            start = max(0.0, float(row["inicio_s"]) - margin_before)
            end = min(duration, float(row["fim_s"]) + margin_after)
            clip_path = cut_clip(st.session_state.video_path, start, end, f"golo_{golo}.mp4")
            st.session_state.clips.append({"golo": golo, "path": clip_path, "inicio_s": start, "fim_s": end})
        st.success(f"{len(st.session_state.clips)} clips criados.")

if st.session_state.clips:
    st.subheader("3️⃣ Biblioteca de clips")
    cols = st.columns(2)
    for i, clip in enumerate(st.session_state.clips):
        with cols[i % 2]:
            st.markdown(f"#### Golo {clip['golo']} — {format_time(clip['inicio_s'])} a {format_time(clip['fim_s'])}")
            st.video(clip["path"])

    st.subheader("4️⃣ Analisar com IA")
    if st.button("Analisar todos os golos com IA", type="primary"):
        if not api_key:
            st.error("Mete a OpenAI API Key na barra lateral ou nos Secrets do Streamlit.")
        else:
            results = []
            progress = st.progress(0)
            for idx, clip in enumerate(st.session_state.clips):
                try:
                    analysis = analyze_clip(clip["path"], api_key, team_color, opponent, match_name, n_frames)
                except Exception as e:
                    analysis = {
                        "tipo_golo": "erro",
                        "finalizacao": "indefinido",
                        "tipo_remate": "indefinido",
                        "passe_anterior": "indefinido",
                        "jogadores_tua_equipa_area": None,
                        "adversarios_area": None,
                        "cor_tua_equipa_confirmada": team_color,
                        "resumo_tatico": f"Erro técnico: {str(e)}",
                        "vulnerabilidade_defensiva": "",
                        "nivel_confianca": 0,
                        "avisos": ["Erro na análise IA"]
                    }
                analysis["golo"] = clip["golo"]
                analysis["inicio"] = format_time(clip["inicio_s"])
                analysis["fim"] = format_time(clip["fim_s"])
                analysis["clip_path"] = clip["path"]
                results.append(analysis)
                progress.progress((idx + 1) / len(st.session_state.clips))
            st.session_state.results = results
            st.success("Análise concluída.")

if st.session_state.results:
    st.divider()
    st.header("📊 Dashboard final")
    df = pd.DataFrame(st.session_state.results)

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Golos analisados", len(df))
    m2.metric("Bola parada", int(df["tipo_golo"].astype(str).str.contains("canto|livre|penálti", case=False, regex=True).sum()))
    m3.metric("Transição", int(df["tipo_golo"].astype(str).str.contains("contra|rápido", case=False, regex=True).sum()))
    m4.metric("Confiança média", round(pd.to_numeric(df["nivel_confianca"], errors="coerce").mean(), 1))

    st.subheader("Tabela de análise")
    st.dataframe(df.drop(columns=["clip_path"], errors="ignore"), use_container_width=True)

    st.download_button("⬇️ Exportar CSV", df.drop(columns=["clip_path"], errors="ignore").to_csv(index=False).encode("utf-8"), "analise_golos.csv", "text/csv")
    st.download_button("⬇️ Exportar JSON", json.dumps(st.session_state.results, ensure_ascii=False, indent=2), "analise_golos.json", "application/json")

    st.subheader("Relatório por golo")
    for row in st.session_state.results:
        with st.expander(f"Golo {row.get('golo')} — {row.get('tipo_golo')} — confiança {row.get('nivel_confianca')}%"):
            if row.get("clip_path"):
                st.video(row["clip_path"])
            st.json({k: v for k, v in row.items() if k != "clip_path"})
else:
    st.info("Fluxo: carregar vídeo → criar lista de golos → ajustar tempos → criar clips → analisar com IA → exportar.")
