"""Azerbaijani Phase 10 UI: launch with streamlit run app/streamlit_app.py."""

import sys
from pathlib import Path

import streamlit as st


def sync_case(session) -> None:
    """
    Stores the required ID and detached CaseState across Streamlit reruns.
    """
    st.session_state.session_id = session.session_id
    st.session_state.case_state = session.case_state


def show_sources(sources: list[dict]) -> None:
    """Renders returned source metadata and text without interpreting HTML."""
    if not sources:
        return
    st.subheader("Hüquqi mənbələr")
    for source in sources:
        number = str(source.get("article_number", ""))
        with st.expander("Maddə " + number):
            st.text(str(source.get("title", "")))
            for name, label in (("section", "Bölmə"), ("chapter", "Fəsil")):
                if source.get(name):
                    st.text(label + ": " + str(source[name]))
            if "page_start" in source and "page_end" in source:
                st.caption(f"Mənbə səhifələri: {source['page_start']}–"
                           f"{source['page_end']}")
            st.text(str(source.get("text", "")))


def main() -> None:
    """Builds one page backed by actual dialogue, QA and drafting services."""
    root = str(Path(__file__).resolve().parents[1])
    if root not in sys.path:
        sys.path.insert(0, root)
    from app.backend import AppSession
    from src.config import settings
    from src.dialogue.slot_schema import (
        DisputeCategory, category_label, slots_for)
    from src.generation.case_prompts import FACT_LABELS_AZ

    st.set_page_config(page_title="Əmək hüququ köməkçisi", page_icon="⚖️",
                       layout="wide", initial_sidebar_state="expanded")
    if "app_session" not in st.session_state:
        st.session_state.app_session = AppSession()
    session = st.session_state.app_session
    sync_case(session)
    st.title("Əmək hüququ köməkçisi")
    st.caption("Azərbaycan Respublikasının Əmək Məcəlləsi üzrə "
               "məlumat və ərizə layihəsi")
    st.warning(settings.LEGAL_DISCLAIMER_AZ)
    with st.sidebar:
        st.header("İş sahəsi")
        mode = st.radio("İstifadə rejimi",
                        ["Sürətli sual", "İşin hazırlanması"],
                        key="mode")
        if st.button("Yeni söhbət", key="new_conversation"):
            session.reset()
            st.session_state.category_choice = None
            sync_case(session)
            st.rerun()
        st.info(settings.LEGAL_DISCLAIMER_AZ)
        st.caption("Sessiya bu brauzer bağlantısına aiddir. "
                   "Şəxsi məlumatları yalnız lazım olduqda daxil edin.")
    if mode == "İşin hazırlanması":
        state = session.case_state
        with st.sidebar:
            st.subheader("Toplanmış məlumatlar")
            if state.dispute_category is not None:
                st.text(category_label(state.dispute_category))
                st.caption("Təsdiqlənib" if state.category_confirmed
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
                st.caption(f"Çatışmayan tələb olunan məlumat: {len(missing)}")
        st.info("Probleminizi təsvir edin, kateqoriyanı təsdiq edin və "
                "növbəti suallara cavab verin. Düzəliş üçün "
                "‘Düzəliş: …’ yazın. Kateqoriyanı aşağıda dəyişə bilərsiniz.")
        for record in state.conversation_history:
            if record.get("role") in ("user", "assistant"):
                with st.chat_message(record["role"]):
                    st.text(record.get("content", ""))
        with st.form("category_form"):
            options = [None] + list(DisputeCategory)
            selected = st.selectbox(
                "Kateqoriya seçimi", options,
                format_func=lambda c: "Kateqoriya seçin" if c is None
                else category_label(c), key="category_choice")
            choose = st.form_submit_button("Kateqoriyanı təsdiq et")
        if choose:
            if selected is None:
                session.notice = ("warning", "Əvvəlcə kateqoriya seçin.")
            else:
                with st.spinner("Kateqoriya təsdiqlənir…"):
                    session.submit_case(category_label(selected), selected)
                sync_case(session)
                st.rerun()
        if state.is_ready_for_draft():
            if st.button("Ərizə layihəsini hazırla", key="generate_complaint"):
                with st.spinner("Mənbələr yoxlanılır və layihə hazırlanır…"):
                    session.generate()
                st.rerun()
        else:
            st.caption("Ərizə üçün kateqoriya təsdiqlənməli və "
                       "tələb olunan məlumatlar tamamlanmalıdır.")
        if (session.complaint is not None
                and session.complaint.draft is not None):
            st.subheader("Ərizə layihəsi")
            st.text(session.complaint.draft)
            st.caption("Təqdim etməzdən əvvəl yoxlayın və "
                       "boş saxlanmış məlumatları tamamlayın.")
        show_sources(state.retrieved_articles
                     if state.category_confirmed else [])
    else:
        st.info("Əmək hüququ ilə bağlı bir sual verin. Cavab yalnız "
                "tapılmış mənbələrə əsaslanacaq. İş haqqında məlumatları "
                "ardıcıl toplamaq üçün ‘İşin hazırlanması’ rejimini seçin.")
        for record in session.quick_history:
            with st.chat_message(record["role"]):
                st.text(record["content"])
        show_sources(session.quick_sources)
    if session.notice is not None:
        kind, text = session.notice
        if kind == "error":
            st.error(text)
        else:
            st.warning(text)
    message = st.chat_input("Əmək probleminizi və ya sualınızı yazın…",
                            key="chat_message")
    if message is not None:
        with st.spinner("Mesajınız emal olunur…"):
            if mode == "İşin hazırlanması":
                session.submit_case(message)
            else:
                session.submit_quick(message)
        sync_case(session)
        st.rerun()


if __name__ == "__main__":
    main()
