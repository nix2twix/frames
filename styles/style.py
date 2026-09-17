import streamlit as st

def load_header():
    st.set_page_config(page_title="FRAMES", layout="wide", page_icon="👾")
    _container = st.container(horizontal=True, vertical_alignment="bottom")
    with _container:
        st.markdown("#### 👾 FRAMES")
        st.markdown("is an interactive tool for test face recognition models")

def load_styles():
    st.markdown("""
        <style>

            /* Markdown (обычный текст) */
            .stMarkdown p,
            .stMarkdown li,
            .stMarkdown div:not(.st-emotion-cache-*) {
                font-size: var(--small-text-size) !important;
                font-weight: var(--small-text-weight) !important;
            }

            /* Текст в виджетах */
            .stSelectbox label,
            .stNumberInput label,
            .stTextInput label,
            .stTextArea label,
            .stDateInput label,
            .stTimeInput label,
            .stCheckbox label,
            .stRadio label {
                font-size: var(--small-text-size) !important;
                font-weight: var(--small-text-weight) !important;
            }

            /* Текст в выпадающих списках и полях ввода */
            .stSelectbox div[data-baseweb="select"] span,
            .stNumberInput input,
            .stTextInput input,
            .stTextArea textarea,
            .stDateInput input,
            .stTimeInput input {
                font-size: var(--small-text-size) !important;
            }

            /* Expander */
            .streamlit-expanderHeader {
                font-size: var(--small-text-size) !important;
                font-weight: var(--small-text-weight) !important;
            }

            /* Metric */
            .stMetric label {
                font-size: var(--small-text-size) !important;
                font-weight: var(--small-text-weight) !important;
            }

            /* Информационные сообщения */
            .stAlert p {
                font-size: var(--small-text-size) !important;
                font-weight: var(--small-text-weight) !important;
            }

            /* Tabs вложенные */
            .stTabs [data-baseweb="tab"]:not([aria-selected="true"]) {
                font-size: var(--small-text-size) !important;
                font-weight: var(--small-text-weight) !important;
            }

            /* Заголовки */
            .stMarkdown h1 {
                font-size: 2rem !important;
                font-weight: 700 !important;
                margin-top: 0rem !important;
                padding-top: 0 !important;
            }
            </style>
    """, unsafe_allow_html=True)