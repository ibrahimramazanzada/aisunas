"""Presentation-only helpers; untrusted legal content never becomes HTML."""

from html import escape
from pathlib import Path

import streamlit as st


SCALES = '''<svg viewBox="0 0 48 48" fill="none" aria-hidden="true"
 xmlns="http://www.w3.org/2000/svg" stroke="currentColor" stroke-width="1.7"
 stroke-linecap="round" stroke-linejoin="round">
 <path d="M24 6v32M12 40h24M18 36h12M9 14h30M24 10l-5 4m5-4 5 4"/>
 <path d="M11 14 5 27h12l-6-13ZM37 14l-6 13h12l-6-13Z"/>
 <path d="M5 27c0 6 12 6 12 0M31 27c0 6 12 6 12 0"/>
 </svg>'''


def apply_styles() -> None:
    """Loads static application CSS, never interpolating user content."""
    css = Path(__file__).with_name("styles.css").read_text(encoding="utf-8")
    st.html("<style>" + css + "</style>")


def welcome() -> None:
    """Shows an Azerbaijani introduction without adding a new brand."""
    st.html('<section class="welcome"><div class="brand-mark">'
            + SCALES + '</div><div class="eyebrow">Əmək hüququ üzrə köməkçi'
            '</div><h2>Hüquqlarınızı anlamağa<br>buradan başlayın.</h2>'
            '<p>Sualınızı yazın. Cavabın əsaslandığı Əmək Məcəlləsi '
            'maddələrini birlikdə nəzərdən keçirin.</p></section>')


def copy_panel(text: str, label: str) -> None:
    """Uses Streamlit's native code-block copy control, with no JavaScript."""
    with st.expander(label):
        st.caption("Mətni köçürmək üçün sağ yuxarıdakı kopyalama düyməsini "
                   "istifadə edin.")
        st.code(text, language=None, wrap_lines=True)


def message(record: dict) -> None:
    """Formats assistant Markdown safely; user messages stay literal text."""
    role = record.get("role")
    if role not in ("user", "assistant"):
        return
    avatar = ":material/person:" if role == "user" else ":material/balance:"
    with st.chat_message(role, avatar=avatar):
        st.html('<span class="message-' + role + '-marker"></span>')
        content = str(record.get("content", ""))
        if role == "assistant":
            st.markdown(escape(content), unsafe_allow_html=False)
        else:
            st.text(content)


def show_sources(sources: list[dict], key: str = "source_evidence") -> None:
    """Displays only actual returned source text and metadata."""
    if not sources:
        return
    with st.container(key=key):
        st.subheader("Hüquqi mənbələr")
        st.caption("Cavabın əsaslandığı mətnləri açıb yoxlaya bilərsiniz.")
        for source in sources:
            with st.expander("Maddə " + str(source.get("article_number", ""))):
                st.text(str(source.get("title", "")))
                for name, label in (("section", "Bölmə"),
                                    ("chapter", "Fəsil")):
                    if source.get(name):
                        st.caption(label + ": " + str(source[name]))
                if "page_start" in source and "page_end" in source:
                    st.caption(f"Mənbə səhifələri: {source['page_start']}–"
                               f"{source['page_end']}")
                st.text(str(source.get("text", "")))
