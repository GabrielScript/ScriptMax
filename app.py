import streamlit as st
import time
import os
import json
import re
from datetime import datetime
from audio_enhancer import AudioEnhancer
from transcriber import Transcriber
from summarizer import Summarizer

st.set_page_config(page_title="MaxClass PDF Generator (DeepSeek)", page_icon="🎤", layout="centered")

# --- Diretório base para relatórios salvos ---
REPORTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "relatorios")
INDEX_FILE = os.path.join(REPORTS_DIR, "index.json")

# --- Inicialização de componentes ---

@st.cache_resource
def get_transcriber_v2():
    return Transcriber()

@st.cache_resource
def get_summarizer():
    return Summarizer()

@st.cache_resource
def get_enhancer():
    return AudioEnhancer()


def _load_report_index():
    """Carrega o índice de relatórios salvos."""
    if os.path.exists(INDEX_FILE):
        with open(INDEX_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    return []


def _save_report_index(index):
    """Salva o índice de relatórios."""
    os.makedirs(REPORTS_DIR, exist_ok=True)
    with open(INDEX_FILE, "w", encoding="utf-8") as f:
        json.dump(index, f, ensure_ascii=False, indent=2)


def _sanitize_name(name):
    """Limpa nome para uso como nome de pasta/arquivo."""
    name = re.sub(r'[<>:"/\\|?*]', '', name)
    name = name.strip()
    return name if name else "Sem_Assunto"


def _save_reports(subject, report_text, summarizer):
    """
    Salva os relatórios HTML e PDF organizados por assunto.
    Retorna (html_path, pdf_path) absolutos.
    """
    safe_subject = _sanitize_name(subject)
    subject_dir = os.path.join(REPORTS_DIR, safe_subject)
    os.makedirs(subject_dir, exist_ok=True)

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M")
    base_name = f"{timestamp}_{safe_subject}"

    html_path = os.path.join(subject_dir, f"{base_name}.html")
    pdf_path = os.path.join(subject_dir, f"{base_name}.pdf")

    summarizer.generate_html_report(report_text, html_path)
    summarizer.generate_pdf(report_text, pdf_path)

    # Atualizar índice
    index = _load_report_index()
    index.append({
        "subject": subject,
        "date": datetime.now().isoformat(),
        "html": html_path,
        "pdf": pdf_path,
        "timestamp": timestamp,
    })
    _save_report_index(index)

    return html_path, pdf_path


def process_file_and_generate_report(audio_path, subject):
    """Pipeline completo: Enhancement → Transcrição → Relatório → Salvamento."""
    enhancer = get_enhancer()
    transcriber = get_transcriber_v2()
    summarizer = get_summarizer()

    total_start = time.time()

    # --- ETAPA 1: Melhoramento do áudio ---
    st.info("🔧 Etapa 1/3 — Melhorando qualidade do áudio (removendo ruído)...")
    t1 = time.time()
    with st.spinner("Aplicando filtros de redução de ruído..."):
        enhanced_path = enhancer.enhance(audio_path)
    t1_elapsed = time.time() - t1

    if enhanced_path != audio_path:
        st.success(f"Áudio melhorado com sucesso! ({t1_elapsed:.1f}s)")
    else:
        st.warning(f"Não foi possível melhorar o áudio. Usando original. ({t1_elapsed:.1f}s)")

    # --- ETAPA 2: Transcrição ---
    st.info("🎤 Etapa 2/3 — Transcrevendo áudio...")
    t2 = time.time()
    with st.spinner("Transcrevendo com faster-whisper..."):
        text = transcriber.transcribe(enhanced_path)
    t2_elapsed = time.time() - t2

    # Limpar arquivo enhanced temporário
    if enhanced_path != audio_path and os.path.exists(enhanced_path):
        try:
            os.remove(enhanced_path)
        except:
            pass

    if not text or text.startswith("Erro"):
        st.error("Falha ao transcrever o áudio.")
        return

    st.success(f"Transcrição concluída! ({t2_elapsed:.1f}s)")
    with st.expander("Ver Transcrição Bruta"):
        st.write(text)

    # --- ETAPA 3: Geração do relatório ---
    st.info("📝 Etapa 3/3 — Gerando relatório com DeepSeek...")
    t3 = time.time()
    with st.spinner("Gerando relatório didático..."):
        report = summarizer.summarize(text)
    t3_elapsed = time.time() - t3

    st.success(f"Relatório gerado! ({t3_elapsed:.1f}s)")
    with st.expander("Ver Relatório", expanded=True):
        st.markdown(report)

    # --- Salvamento organizado por assunto ---
    with st.spinner("Salvando relatórios organizados..."):
        html_path, pdf_path = _save_reports(subject, report, summarizer)

    total_elapsed = time.time() - total_start

    st.success(f"✅ Relatórios salvos em `relatorios/{_sanitize_name(subject)}/`")

    # --- Métricas de tempo ---
    st.divider()
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("🔧 Denoising", f"{t1_elapsed:.0f}s")
    m2.metric("🎤 Transcrição", f"{t2_elapsed:.0f}s")
    m3.metric("🤖 DeepSeek", f"{t3_elapsed:.0f}s")
    m4.metric("⏱️ Total", f"{total_elapsed:.0f}s")

    # Botões de download lado a lado
    col1, col2 = st.columns(2)

    with col1:
        with open(pdf_path, "rb") as pdf_file:
            st.download_button(
                label="📄 Baixar PDF (com fórmulas)",
                data=pdf_file,
                file_name=os.path.basename(pdf_path),
                mime="application/pdf",
                type="primary"
            )
        st.caption("✨ PDF com fórmulas renderizadas")

    with col2:
        with open(html_path, "rb") as html_file:
            st.download_button(
                label="📐 Baixar HTML",
                data=html_file,
                file_name=os.path.basename(html_path),
                mime="text/html"
            )
        st.caption("Texto simples")

    st.balloons()


# ============================================================
# SIDEBAR — Relatórios Salvos
# ============================================================
with st.sidebar:
    st.header("📚 Relatórios Salvos")

    index = _load_report_index()
    if not index:
        st.info("Nenhum relatório salvo ainda.\nProcesse um áudio para começar!")
    else:
        # Agrupar por assunto
        subjects = {}
        for entry in index:
            subj = entry.get("subject", "Sem Assunto")
            if subj not in subjects:
                subjects[subj] = []
            subjects[subj].append(entry)

        for subj, entries in sorted(subjects.items()):
            with st.expander(f"📁 {subj} ({len(entries)})", expanded=False):
                for entry in sorted(entries, key=lambda x: x["date"], reverse=True):
                    date_str = entry.get("timestamp", "")
                    col_a, col_b = st.columns(2)

                    html_exists = os.path.exists(entry.get("html", ""))
                    pdf_exists = os.path.exists(entry.get("pdf", ""))

                    st.caption(f"📅 {date_str}")
                    if html_exists:
                        with open(entry["html"], "rb") as f:
                            st.download_button(
                                f"📐 HTML",
                                data=f,
                                file_name=os.path.basename(entry["html"]),
                                mime="text/html",
                                key=f"sidebar_html_{entry['date']}",
                            )
                    if pdf_exists:
                        with open(entry["pdf"], "rb") as f:
                            st.download_button(
                                f"📄 PDF",
                                data=f,
                                file_name=os.path.basename(entry["pdf"]),
                                mime="application/pdf",
                                key=f"sidebar_pdf_{entry['date']}",
                            )
                    st.divider()


# ============================================================
# CONTEÚDO PRINCIPAL
# ============================================================
st.title("🎓 MaxClass PDF Generator")
st.write("Grave sua aula ou faça upload de um áudio. A IA irá **melhorar o áudio**, transcrever e gerar um relatório perfeito.")

# Campo obrigatório de assunto
subject = st.text_input(
    "📌 Assunto da aula (usado para organizar seus relatórios)",
    placeholder="Ex: Cálculo Integral, Física Quântica, História do Brasil...",
)

if not subject:
    st.warning("⬆️ Informe o assunto da aula antes de processar.")

tab1, tab2 = st.tabs(["🎙️ Gravar Aula", "📁 Fazer Upload de Áudio"])

# --- TAB 1: GRAVAR AULA ---
with tab1:
    st.header("Gravar Aula ao Vivo")
    st.write("Clique no botão abaixo para começar a gravar usando o microfone do seu navegador/celular.")

    from streamlit_mic_recorder import mic_recorder
    
    if not subject:
        st.info("⬆️ Preencha o assunto acima para habilitar o gravador.")
    else:
        # A key is important so that streamlit-mic-recorder knows which instance it is
        audio = mic_recorder(
            start_prompt="🔴 Iniciar Gravação",
            stop_prompt="⏹️ Parar Gravação",
            format="wav",
            key='recorder'
        )

        if audio:
            # streamlit run reruns the app when the component updates.
            # We want to process only if this is a new recording.
            audio_id = audio.get('id', '')
            
            if st.session_state.get('last_recorded_audio_id') != audio_id:
                st.info("Gravação recebida! Salvando arquivo temporário...")
                
                # Extract bytes and save to a temporary WAV file
                temp_rec_path = f"temp_recorded_{audio_id}.wav"
                with open(temp_rec_path, "wb") as f:
                    f.write(audio['bytes'])
                
                # Check if file has some size
                if os.path.exists(temp_rec_path) and os.path.getsize(temp_rec_path) > 100:
                    process_file_and_generate_report(temp_rec_path, subject)
                else:
                    st.error("A gravação parece estar vazia ou falhou.")
                
                # Clean up the temporay file
                if os.path.exists(temp_rec_path):
                    try:
                        os.remove(temp_rec_path)
                    except Exception as e:
                        pass
                
                st.session_state['last_recorded_audio_id'] = audio_id
            else:
                st.success("✔ Gravação processada! (Grave novamente ou mude o assunto)")


# --- TAB 2: UPLOAD DE ÁUDIO ---
with tab2:
    st.header("Upload de Áudio Existente")
    uploaded_file = st.file_uploader("Envie seu arquivo de áudio (wav, mp3, m4a...)", type=["wav", "mp3", "m4a", "ogg"])

    if uploaded_file is not None and subject:
        file_id = f"{uploaded_file.name}_{uploaded_file.size}_{subject}"

        if st.session_state.get('last_processed_file') != file_id:
            temp_audio_path = f"temp_{uploaded_file.name}"
            with open(temp_audio_path, "wb") as f:
                f.write(uploaded_file.getbuffer())

            process_file_and_generate_report(temp_audio_path, subject)

            if os.path.exists(temp_audio_path):
                try:
                    os.remove(temp_audio_path)
                except:
                    pass

            st.session_state['last_processed_file'] = file_id
        else:
            st.success(f"✔ O arquivo '{uploaded_file.name}' já foi processado! Envie um novo áudio ou altere o assunto.")
    elif uploaded_file is not None and not subject:
        st.warning("⬆️ Informe o assunto da aula acima antes de processar.")
