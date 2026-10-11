"""Charcoal and gold Streamlit interface over the existing legal pipelines."""

import sys
from pathlib import Path

import streamlit as st


EXAMPLES = (
    ("İşdən çıxarılma", "İşəgötürən əmək müqaviləsinə hansı əsaslarla "
     "xitam verə bilər?"),
    ("Əmək haqqı", "Əmək haqqım vaxtında ödənilmirsə, hansı "
     "müddəalara baxmalıyam?"),
    ("Məzuniyyət", "İllik əmək məzuniyyəti necə tənzimlənir?"),
    ("Əmək müqaviləsi", "Əmək müqaviləsində hansı məlumatlar olmalıdır?"),
)


def sync_case(session) -> None:
    """Keeps detached case state and the private session ID across reruns."""
    st.session_state.session_id = session.session_id
    st.session_state.case_state = session.case_state


def main() -> None:
    """Renders real chat, case confirmation, evidence and validated drafts."""
    root = str(Path(__file__).resolve().parents[1])
    if root not in sys.path:
        sys.path.insert(0, root)
    from app.backend import AppSession
    from app.presentation import (
        SCALES, apply_styles, copy_panel, message, show_sources, welcome,
    )
    from src.config import settings
    from src.dialogue.slot_schema import (
        DisputeCategory, category_label, required_slots, slots_for,
    )
    from src.generation.case_prompts import FACT_LABELS_AZ

    st.set_page_config(page_title="Əmək hüququ köməkçisi",
                       page_icon=":material/balance:", layout="wide",
                       initial_sidebar_state="auto")
    apply_styles()
    if "app_session" not in st.session_state:
        st.session_state.app_session = AppSession()
    session = st.session_state.app_session
    sync_case(session)
    with st.sidebar:
        st.html('<div class="brand-mark">' + SCALES + '</div>'
                '<div class="brand-label">az-legal-rag</div>')
        st.subheader("Əmək hüququ köməkçisi")
        st.caption("Azərbaycan Respublikasının Əmək Məcəlləsi")
        st.divider()
        mode = st.radio("İstifadə rejimi",
                        ["Sürətli sual", "İşin hazırlanması"], key="mode")
        if st.button("Yeni söhbət", key="new_conversation",
                     use_container_width=True, icon=":material/add:"):
            if mode == "Sürətli sual":
                session.reset_quick()
            else:
                session.reset_case()
                st.session_state.category_choice = None
            sync_case(session)
            st.rerun()
        st.caption("Yeni söhbət yalnız seçilmiş rejimin məlumatlarını "
                   "təmizləyir.")
        with st.expander("Necə istifadə etməli?"):
            st.write("Sürətli sual rejimində hüquqi sualınızı yazın və "
                     "mənbələri yoxlayın. İşin hazırlanması rejimində "
                     "kateqoriyanı təsdiqləyib suallara cavab verin.")
            st.write("Məlumatı dəyişmək üçün ‘Düzəliş: …’ yazın.")
        st.divider()
        st.caption("Söhbətlər yalnız cari sessiyada saxlanılır. "
                   "Şəxsi məlumatları yalnız lazım olduqda daxil edin.")
        st.caption(settings.LEGAL_DISCLAIMER_AZ)

    st.title("Əmək hüququ köməkçisi")
    st.caption(settings.LEGAL_DISCLAIMER_AZ)
    if mode == "Sürətli sual":
        if not session.quick_history:
            welcome()
            st.caption("Bir mövzu seçin və ya aşağıda öz sualınızı yazın.")
            columns = st.columns(2)
            for i, (label, question) in enumerate(EXAMPLES):
                with columns[i % 2]:
                    if st.button(label, key=f"example_{i}",
                                 use_container_width=True):
                        with st.spinner("Mənbələr yoxlanılır və cavab "
                                        "hazırlanır…"):
                            session.submit_quick(question)
                        st.rerun()
        for i, record in enumerate(session.quick_history):
            message(record)
            if record["role"] == "assistant":
                sources = record.get("sources", [])
                if not sources and i == len(session.quick_history) - 1:
                    sources = session.quick_sources
                show_sources(sources, key=f"source_evidence_{i}")
                copy_panel(record["content"], "Cavabı kopyala")
    else:
        state = session.case_state
        st.subheader("İşinizi addım-addım hazırlayaq")
        st.caption("Kateqoriyanı təsdiqləyin → məlumatları tamamlayın → "
                   "mənbələrə əsaslanan layihəni yoxlayın.")
        requirements = (required_slots(state.dispute_category)
                        if state.dispute_category else [])
        filled = sum(name in state.filled_slots for name in requirements)
        fraction = (filled / len(requirements)
                    if requirements and state.category_confirmed else 0)
        st.progress(fraction, text=(
            f"Tələb olunan məlumatlar: {filled}/{len(requirements)}"
            if state.category_confirmed else "Kateqoriya təsdiqi gözlənilir"))
        with st.sidebar:
            st.subheader("İşin xülasəsi")
            if state.dispute_category:
                st.text(category_label(state.dispute_category))
                st.caption("Kateqoriya təsdiqlənib" if state.category_confirmed
                           else "Təklifdir — təsdiq gözlənilir")
            else:
                st.caption("Kateqoriya hələ seçilməyib.")
            for name in (slots_for(state.dispute_category)
                         if state.dispute_category else []):
                if name in state.filled_slots:
                    st.text(FACT_LABELS_AZ[name] + ": "
                            + str(state.filled_slots[name]))
            if state.category_confirmed:
                missing = state.missing_required_slots()
                with st.expander(f"Çatışmayan məlumatlar · {len(missing)}"):
                    if missing:
                        for name in missing:
                            st.text(FACT_LABELS_AZ[name])
                    else:
                        st.caption("Bütün tələb olunan məlumatlar toplanıb.")
        with st.expander("Kateqoriyanı təsdiqləyin və ya dəyişin",
                         expanded=not state.category_confirmed):
            with st.form("category_form"):
                selected = st.selectbox(
                    "Kateqoriya seçimi", [None] + list(DisputeCategory),
                    format_func=lambda c: "Kateqoriya seçin" if c is None
                    else category_label(c), key="category_choice")
                choose = st.form_submit_button("Kateqoriyanı təsdiq et",
                                               use_container_width=True)
            if choose:
                if selected is None:
                    session.notice = ("warning", "Əvvəlcə kateqoriya seçin.")
                else:
                    with st.spinner("Kateqoriya təsdiqlənir…"):
                        session.submit_case(category_label(selected), selected)
                    sync_case(session)
                    st.rerun()
        if not state.conversation_history:
            st.info("Əmək probleminizi təsvir edin. Kateqoriya təsdiqindən "
                    "sonra sizə çatışmayan məlumatlar barədə suallar "
                    "veriləcək.")
        for record in state.conversation_history:
            message(record)
        if state.is_ready_for_draft():
            st.success("Tələb olunan məlumatlar tamamdır. Layihə "
                       "hazırlanarkən hüquqi mənbələr ayrıca yoxlanılacaq.")
            if st.button("Ərizə layihəsini hazırla", key="generate_complaint",
                         type="primary", use_container_width=True):
                with st.spinner("Mənbələr yoxlanılır və layihə hazırlanır…"):
                    session.generate()
                st.rerun()
        else:
            st.caption("Layihə üçün kateqoriya təsdiqlənməli və tələb "
                       "olunan məlumatlar tamamlanmalıdır.")
        if session.complaint is not None and session.complaint.draft:
            st.subheader("Ərizə layihəsi")
            with st.container(key="complaint_document"):
                st.text(session.complaint.draft)
            st.caption("Bu, təqdimata hazır hüquqi sənəd deyil. "
                       "Məlumatları və mənbələrin tətbiqini yoxlayın.")
            copy_panel(session.complaint.draft, "Layihəni kopyala")
            st.download_button("Mətn faylını yüklə", session.complaint.draft,
                               file_name="erize-layihesi.txt",
                               mime="text/plain", key="download_complaint")
        show_sources(state.retrieved_articles
                     if state.category_confirmed else [])
    if session.notice:
        kind, text = session.notice
        if kind == "error":
            st.error(text)
        else:
            st.warning(text)
    prompt = ("Əmək hüququ ilə bağlı sualınızı yazın…"
              if mode == "Sürətli sual"
              else "Cavabınızı və ya düzəlişi yazın…")
    incoming = st.chat_input(prompt, key="chat_message")
    if incoming is not None:
        loading = ("Mənbələr yoxlanılır və cavab hazırlanır…"
                   if mode == "Sürətli sual" else "Məlumatlar emal olunur…")
        with st.spinner(loading):
            if mode == "Sürətli sual":
                session.submit_quick(incoming)
            else:
                session.submit_case(incoming)
        sync_case(session)
        st.rerun()


if __name__ == "__main__":
    main()
