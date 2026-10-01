import html
import json
import os
import re
import pandas as pd
import streamlit as st

import tiktoken

# Import chunking methods
from chunking import (
    fixed_char,
    html as html_chunker,
    markdown as markdown_chunker,
    sentence,
)
from chunking.lumberchunker import lumberchunker
from chunking.semantic import clustered, openai_embedding, semantic, tiled

@st.cache_resource
def get_tokenizer():
    try:
        return tiktoken.get_encoding("cl100k_base")
    except Exception:
        return None

def count_tokens(text: str) -> int:
    enc = get_tokenizer()
    if enc is not None:
        return len(enc.encode(text))
    return max(1, round(len(text) / 4.0))

# ---------------------------------------------------------
# Page Configuration & Styling
# ---------------------------------------------------------
st.set_page_config(
    page_title="Chunk Visualizer",
    page_icon="✂️",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    .chunk-badge {
        display: inline-block;
        font-size: 0.72rem;
        font-weight: 700;
        padding: 1px 6px;
        border-radius: 4px;
        margin-right: 4px;
        vertical-align: middle;
        user-select: none;
    }
    .chunk-highlight {
        border-radius: 4px;
        padding: 2px 4px;
        margin: 1px 0;
        display: inline;
        line-height: 1.8;
    }
    .chunk-card {
        border: 1px solid #e0e0e0;
        border-radius: 8px;
        padding: 14px;
        margin-bottom: 12px;
        background-color: #fafafa;
    }
    .metric-container {
        border-radius: 8px;
        padding: 10px 14px;
        background-color: #f7f9fc;
        border: 1px solid #e3e8ee;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------
# Color Palette for Chunks
# ---------------------------------------------------------
PALETTE = [
    {"bg": "#E3F2FD", "border": "#90CAF9", "text": "#1565C0", "badge_bg": "#1976D2", "badge_text": "#FFFFFF"},
    {"bg": "#E8F5E9", "border": "#A5D6A7", "text": "#2E7D32", "badge_bg": "#388E3C", "badge_text": "#FFFFFF"},
    {"bg": "#FFF3E0", "border": "#FFCC80", "text": "#E65100", "badge_bg": "#F57C00", "badge_text": "#FFFFFF"},
    {"bg": "#F3E5F5", "border": "#CE93D8", "text": "#7B1FA2", "badge_bg": "#8E24AA", "badge_text": "#FFFFFF"},
    {"bg": "#E0F7FA", "border": "#80DEEA", "text": "#00838F", "badge_bg": "#0097A7", "badge_text": "#FFFFFF"},
    {"bg": "#FFFDE7", "border": "#FFF59D", "text": "#F57F17", "badge_bg": "#FBC02D", "badge_text": "#212121"},
    {"bg": "#FCE4EC", "border": "#F48FB1", "text": "#C2185B", "badge_bg": "#D81B60", "badge_text": "#FFFFFF"},
    {"bg": "#EDE7F6", "border": "#B39DDB", "text": "#512DA8", "badge_bg": "#5E35B1", "badge_text": "#FFFFFF"},
    {"bg": "#EFEBE9", "border": "#BCAAA4", "text": "#4E342E", "badge_bg": "#5D4037", "badge_text": "#FFFFFF"},
    {"bg": "#E0F2F1", "border": "#80CBC4", "text": "#00695C", "badge_bg": "#00796B", "badge_text": "#FFFFFF"},
]

def get_color(index: int):
    return PALETTE[index % len(PALETTE)]

# ---------------------------------------------------------
# Sample Texts
# ---------------------------------------------------------
SAMPLE_MARKDOWN = """# Retrieval-Augmented Generation (RAG) Architecture

Retrieval-Augmented Generation (RAG) combines the strengths of pre-trained language models with external retrieval mechanisms. By grounding responses in verified knowledge sources, RAG mitigates hallucinations and keeps outputs up to date without continuous fine-tuning.

## Indexation Pipeline

The indexation phase prepares raw documents for efficient and accurate retrieval. It consists of multiple stages:

1. **Document Loading**: Ingesting heterogeneous formats such as PDF, HTML, Markdown, and raw text.
2. **Chunking**: Partitioning long documents into self-contained segments. The chunking strategy directly influences retrieval precision and contextual completeness.
3. **Embedding**: Converting text chunks into high-dimensional dense vector representations.
4. **Vector Indexing**: Storing vectors into an Approximate Nearest Neighbor (ANN) index like HNSW or FAISS.

## Chunking Strategies

Selecting an optimal chunking strategy depends on document structure and the downstream task:

### Fixed-Size Window Chunking
Fixed-size chunking slices text into segments of a determined character or token count, typically with a sliding overlap. While straightforward and fast, it risks cutting through sentences or coherent ideas.

### Structure-Aware Chunking
Preserves existing semantic boundaries like Markdown headings, HTML DOM tags, or natural sentence/paragraph delimiters. This ensures that related sentences remain intact and headings provide contextual anchoring.

### Semantic & Embedding-Based Chunking
Calculates embedding similarity across adjacent sentences or blocks to detect genuine topic transitions. Cuts are placed where semantic similarity drops below a calculated percentile threshold.

## Retrieval and Reranking

Once indexed, queries are matched against chunks using vector similarity or hybrid BM25 + dense search. A cross-encoder reranker further refines top candidates to maximize relevance before final generation.
"""

SAMPLE_PROSE = """Artificial intelligence and machine learning have undergone unprecedented transformation over the past decade. Early statistical models relied heavily on manual feature engineering and domain-specific heuristics. As computational power expanded with specialized GPUs and distributed computing frameworks, deep neural networks became practical for large-scale perception tasks such as computer vision and speech recognition.

The breakthrough of transformer architectures in 2017 revolutionized natural language processing. Self-attention mechanisms enabled models to process entire sequences in parallel, capturing long-range dependencies far more effectively than recurrent architectures. Subsequent models demonstrated emergent reasoning, few-shot learning, and sophisticated language understanding.

However, large language models face inherent limitations. Their internal knowledge is fixed at pre-training time, making them vulnerable to obsolescence and hallucinations when queried on private or recent information. Retrieval-augmented architectures address this gap by dynamically fetching pertinent external passages before generating answers.

Evaluation of such systems requires measuring both retrieval accuracy and generation fidelity. Retrieval metrics include Mean Reciprocal Rank (MRR), Normalized Discounted Cumulative Gain (NDCG), and chunk span recall. Generation metrics evaluate answer relevance, factual consistency, and absence of hallucinations relative to the retrieved context.
"""

SAMPLE_HTML = """<!DOCTYPE html>
<html>
<head><title>System Documentation</title></head>
<body>
<h1>System Architecture Overview</h1>
<p>This document details the microservices architecture, data ingestion pipelines, and caching layers supporting our high-throughput data processing cluster.</p>
<h2>Service Components</h2>
<p>The cluster is partitioned into stateless API gateways, message queues, and worker pools communicating over gRPC.</p>
<p>Each service maintains independent autoscaling policies based on queue latency and CPU utilization metrics.</p>
<h2>Data Pipeline & Storage</h2>
<p>Raw telemetry is ingested through distributed event streams and processed using real-time stream analytics.</p>
<p>Aggregated records are written to a primary document database, while high-frequency metrics are piped directly to a time-series store.</p>
<h3>Security & Compliance</h3>
<p>All inter-service traffic is encrypted using mutual TLS (mTLS) with automated certificate rotation every thirty days.</p>
</body>
</html>"""

SAMPLE_POLISH = """Gdańsk jest jednym z najstarszych miast w Polsce o ponadtysiącletniej historii. Położony nad Zatoką Gdańską u ujścia Motławy do Wisły, stanowi centrum kulturalne, naukowe i gospodarcze północnej Polski.

W średniowieczu miasto należało do Hanzy, związku miast handlowych Europy Północnej. Dzięki handlowi zbożem i drewnem Gdańsk stał się najbogatszym miastem Rzeczypospolitej Obojga Narodów, zyskując przydomek „spichlerza Europy”.

W XX wieku Gdańsk odegrał kluczową rolę w historii współczesnej. Na Westerplatte rozpoczęła się II wojna światowa w Europie. Cztery dekady później w Stoczni Gdańskiej narodził się Niezależny Samorządny Związek Zawodowy „Solidarność”, który przyczynił się do upadku komunizmu w Europie Środkowo-Wschodniej.
"""

# ---------------------------------------------------------
# Sidebar Controls
# ---------------------------------------------------------
st.sidebar.title("✂️ Chunk Visualizer")
st.sidebar.caption("Experiment with different chunking algorithms and inspect results.")

# Preset Selector
preset_choice = st.sidebar.selectbox(
    "Load Preset Sample Text:",
    options=["Custom Input", "Markdown Technical Doc", "Prose / Technical Paper", "HTML Source", "Polish Text (PoQuAD)"],
    index=1,
)

if "current_preset" not in st.session_state:
    st.session_state.current_preset = preset_choice
    st.session_state.input_text = SAMPLE_MARKDOWN

if preset_choice != st.session_state.current_preset:
    st.session_state.current_preset = preset_choice
    if preset_choice == "Markdown Technical Doc":
        new_text = SAMPLE_MARKDOWN
    elif preset_choice == "Prose / Technical Paper":
        new_text = SAMPLE_PROSE
    elif preset_choice == "HTML Source":
        new_text = SAMPLE_HTML
    elif preset_choice == "Polish Text (PoQuAD)":
        new_text = SAMPLE_POLISH
    else:
        new_text = ""
    st.session_state.input_text = new_text
    st.session_state["main_text_input"] = new_text

# Chunking Method Selection
st.sidebar.subheader("Chunking Method")
method = st.sidebar.selectbox(
    "Select Method:",
    options=[
        "Fixed Character (fixed_char)",
        "Structure / Sentence (sentence)",
        "Markdown Outline (markdown)",
        "HTML Outline (html)",
        "Semantic Breakpoint (semantic)",
        "Semantic Tiled / TextTiling (tiled)",
        "Semantic Clustered (clustered)",
        "LumberChunker (LLM-based)",
    ],
    index=1,
)

# Method Parameters
st.sidebar.subheader("Parameters")
params = {}

if method == "Fixed Character (fixed_char)":
    params["size"] = st.sidebar.number_input("Window Size (characters):", min_value=10, max_value=10000, value=500, step=50)
    max_overlap = max(0, params["size"] - 1)
    params["overlap"] = st.sidebar.number_input("Overlap (characters):", min_value=0, max_value=max_overlap, value=min(100, max_overlap), step=10)

elif method == "Structure / Sentence (sentence)":
    params["max_tokens"] = st.sidebar.number_input("Max Tokens (per chunk):", min_value=10, max_value=4096, value=256, step=16)
    max_overlap = max(0, params["max_tokens"] - 1)
    params["overlap"] = st.sidebar.number_input("Overlap Tokens:", min_value=0, max_value=max_overlap, value=min(40, max_overlap), step=5)
    params["language"] = st.sidebar.selectbox("Language (pysbd code):", options=["auto", "en", "pl", "fr", "de", "es"], index=0)

elif method == "Markdown Outline (markdown)":
    split_subsections = st.sidebar.checkbox("Split large sections by token budget", value=True)
    if split_subsections:
        params["max_tokens"] = st.sidebar.number_input("Max Tokens:", min_value=10, max_value=4096, value=256, step=16)
    else:
        params["max_tokens"] = None
    params["language"] = st.sidebar.selectbox("Language (pysbd code):", options=["auto", "en", "pl", "fr", "de", "es"], index=0)

elif method == "HTML Outline (html)":
    split_subsections = st.sidebar.checkbox("Split large sections by token budget", value=True)
    if split_subsections:
        params["max_tokens"] = st.sidebar.number_input("Max Tokens:", min_value=10, max_value=4096, value=256, step=16)
    else:
        params["max_tokens"] = None

elif method == "Semantic Breakpoint (semantic)":
    params["breakpoint_percentile"] = st.sidebar.slider("Breakpoint Percentile Threshold:", min_value=50, max_value=99, value=95)
    params["buffer_size"] = st.sidebar.slider("Buffer Size (adjacent sentences):", min_value=1, max_value=5, value=1)
    params["language"] = st.sidebar.selectbox("Language (pysbd code):", options=["auto", "en", "pl", "fr", "de", "es"], index=0)

elif method == "Semantic Tiled / TextTiling (tiled)":
    params["block"] = st.sidebar.slider("Block Size (sentences per window):", min_value=1, max_value=10, value=3)
    params["percentile"] = st.sidebar.slider("Dissimilarity Percentile:", min_value=50, max_value=99, value=90)
    params["neighbourhood"] = st.sidebar.slider("Neighbourhood Filter:", min_value=0, max_value=5, value=2)
    params["language"] = st.sidebar.selectbox("Language (pysbd code):", options=["auto", "en", "pl", "fr", "de", "es"], index=0)

elif method == "Semantic Clustered (clustered)":
    params["max_tokens"] = st.sidebar.number_input("Max Tokens:", min_value=20, max_value=4096, value=512, step=32)
    params["percentile"] = st.sidebar.slider("Distance Percentile Cut:", min_value=50, max_value=99, value=95)
    params["buffer"] = st.sidebar.slider("Buffer Size:", min_value=1, max_value=5, value=1)
    params["language"] = st.sidebar.selectbox("Language (pysbd code):", options=["auto", "en", "pl", "fr", "de", "es"], index=0)

elif method == "LumberChunker (LLM-based)":
    params["model"] = st.sidebar.text_input("LLM Model Name:", value=os.environ.get("LLM_MODEL", ""))

# OpenAI API Settings for embedding/LLM methods
needs_openai = "Semantic" in method or "LumberChunker" in method
if needs_openai:
    with st.sidebar.expander("🔑 OpenAI / Embedding Endpoint", expanded=True):
        st.caption("Required for semantic embeddings and LumberChunker.")
        api_key = st.text_input("API Key:", value=os.environ.get("OPENAI_API_KEY", ""), type="password")
        api_base = st.text_input("API Base URL:", value=os.environ.get("OPENAI_API_BASE", ""))
        embedding_model = st.text_input("Embedding Model:", value=os.environ.get("EMBEDDING_MODEL", ""))
        params["api_key"] = api_key if api_key else None
        params["api_base"] = api_base if api_base else None
        params["model_name"] = embedding_model

# ---------------------------------------------------------
# Main Application Layout
# ---------------------------------------------------------
st.title("✂️ Chunk Visualizer")
st.markdown("Inspect and compare how text is partitioned across different chunking strategies.")

# Text Input Area
col_input, col_meta = st.columns([4, 1])
with col_input:
    text_input = st.text_area(
        "Input Document:",
        value=st.session_state.get("input_text", SAMPLE_MARKDOWN),
        height=260,
        placeholder="Paste text, markdown, or HTML here...",
        key="main_text_input",
    )
    # Sync with session state
    st.session_state.input_text = text_input

with col_meta:
    st.markdown("#### Document Info")
    char_len = len(text_input)
    word_count = len(text_input.split())
    token_count = count_tokens(text_input) if text_input else 0
    line_count = len(text_input.splitlines())
    st.metric("Total Characters", f"{char_len:,}")
    st.metric("Tokens (tiktoken)", f"{token_count:,}")
    st.metric("Word Count", f"{word_count:,}")
    st.metric("Line Count", f"{line_count:,}")

# ---------------------------------------------------------
# Computation Function
# ---------------------------------------------------------
def run_chunker(doc_text, method_name, method_params):
    if not doc_text.strip():
        return []
    
    if method_name == "Fixed Character (fixed_char)":
        return fixed_char(doc_text, size=method_params["size"], overlap=method_params["overlap"])
    
    elif method_name == "Structure / Sentence (sentence)":
        return sentence(
            doc_text,
            max_tokens=method_params["max_tokens"],
            overlap=method_params["overlap"],
            language=method_params["language"],
        )
    
    elif method_name == "Markdown Outline (markdown)":
        return markdown_chunker(
            doc_text,
            max_tokens=method_params["max_tokens"],
            language=method_params["language"],
        )
    
    elif method_name == "HTML Outline (html)":
        return html_chunker(
            doc_text,
            max_tokens=method_params["max_tokens"],
        )
    
    elif method_name == "Semantic Breakpoint (semantic)":
        embed_client = openai_embedding(
            model=method_params["model_name"],
            api_key=method_params["api_key"],
            api_base=method_params["api_base"],
        )
        return semantic(
            doc_text,
            embed_model=embed_client,
            breakpoint_percentile=method_params["breakpoint_percentile"],
            buffer_size=method_params["buffer_size"],
            language=method_params["language"],
        )
    
    elif method_name == "Semantic Tiled / TextTiling (tiled)":
        embed_client = openai_embedding(
            model=method_params["model_name"],
            api_key=method_params["api_key"],
            api_base=method_params["api_base"],
        )
        return tiled(
            doc_text,
            embed_model=embed_client,
            block=method_params["block"],
            percentile=method_params["percentile"],
            neighbourhood=method_params["neighbourhood"],
            language=method_params["language"],
        )
    
    elif method_name == "Semantic Clustered (clustered)":
        embed_client = openai_embedding(
            model=method_params["model_name"],
            api_key=method_params["api_key"],
            api_base=method_params["api_base"],
        )
        return clustered(
            doc_text,
            embed_model=embed_client,
            percentile=method_params["percentile"],
            max_tokens=method_params["max_tokens"],
            buffer=method_params["buffer"],
            language=method_params["language"],
        )
    
    elif method_name == "LumberChunker (LLM-based)":
        from llm import get_client
        client = get_client(
            api_key=method_params["api_key"],
            api_base=method_params["api_base"],
        )
        return lumberchunker(
            doc_text,
            model=method_params["model"],
            client=client,
        )
    
    return []

# Execute Chunking
chunks = []
error_message = None

if text_input.strip():
    with st.spinner(f"Computing chunks using {method}..."):
        try:
            chunks = run_chunker(text_input, method, params)
        except Exception as e:
            error_message = str(e)

if error_message:
    st.error(f"⚠️ Error computing chunks: {error_message}")
    if needs_openai and ("api_key" in error_message.lower() or "connection" in error_message.lower() or "401" in error_message or "404" in error_message):
        st.info("Tip: Ensure your OpenAI/embedding API key and base URL are configured in the sidebar.")

# ---------------------------------------------------------
# Visualization & Metrics
# ---------------------------------------------------------
if chunks:
    st.divider()
    st.subheader("📊 Chunking Summary")

    total_chunks = len(chunks)
    chunk_lengths = [len(c.text) for c in chunks]
    chunk_words = [len(c.text.split()) for c in chunks]
    chunk_tokens = [count_tokens(c.text) for c in chunks]

    # Metrics Row
    m_col1, m_col2, m_col3, m_col4, m_col5 = st.columns(5)
    with m_col1:
        st.metric("Total Chunks", total_chunks)
    with m_col2:
        avg_len = sum(chunk_lengths) / total_chunks if total_chunks else 0
        st.metric("Avg Chars", f"{avg_len:.1f}")
    with m_col3:
        min_len, max_len = (min(chunk_lengths), max(chunk_lengths)) if total_chunks else (0, 0)
        st.metric("Min / Max Chars", f"{min_len} / {max_len}")
    with m_col4:
        avg_tokens = sum(chunk_tokens) / total_chunks if total_chunks else 0
        min_tok, max_tok = (min(chunk_tokens), max(chunk_tokens)) if total_chunks else (0, 0)
        st.metric("Avg Tokens (tiktoken)", f"{avg_tokens:.1f}", help=f"Min: {min_tok}, Max: {max_tok}")
    with m_col5:
        # Check overlaps
        has_overlap = any(
            chunks[i].start < chunks[i - 1].end
            for i in range(1, len(chunks))
        )
        st.metric("Chunk Overlap", "Detected" if has_overlap else "None (Disjoint)")

    # Tabs for visualization
    tab_doc, tab_cards, tab_analytics = st.tabs([
        "🎨 Document Highlighting",
        "🗂️ Chunk Inspector",
        "📈 Size Distribution & Export",
    ])

    # -----------------------------------------------------
    # Tab 1: Document Highlighting
    # -----------------------------------------------------
    with tab_doc:
        st.markdown("#### Highlighted Chunks")
        for c in chunks:
            color = get_color(c.index)
            tokens = count_tokens(c.text)
            meta_str = f" | {json.dumps(c.metadata)}" if c.metadata else ""
            badge_html = (
                f"<span class='chunk-badge' style='background-color: {color['badge_bg']}; color: {color['badge_text']}; font-size: 0.8rem; padding: 2px 8px;'>"
                f"Chunk #{c.index} &nbsp; [{c.start}:{c.end}] &nbsp; ({len(c.text)} chars, {tokens} tokens){html.escape(meta_str)}</span>"
            )
            escaped_text = html.escape(c.text)
            st.markdown(
                f"<div style='margin-bottom: 12px; padding: 12px 16px; border-radius: 6px; background-color: {color['bg']}; "
                f"border-left: 6px solid {color['badge_bg']}; border-top: 1px solid {color['border']}; border-right: 1px solid {color['border']}; border-bottom: 1px solid {color['border']};'>"
                f"<div style='margin-bottom: 8px;'>{badge_html}</div>"
                f"<div style='white-space: pre-wrap; font-family: sans-serif; font-size: 0.95rem; color: #212121; line-height: 1.6;'>{escaped_text}</div>"
                f"</div>",
                unsafe_allow_html=True,
            )

    # -----------------------------------------------------
    # Tab 2: Chunk Inspector
    # -----------------------------------------------------
    with tab_cards:
        st.markdown("#### Individual Chunk Breakdown")
        filter_col, sort_col = st.columns([3, 1])
        with filter_col:
            search_query = st.text_input("Filter chunks by text content:", placeholder="Search keywords...")
        with sort_col:
            sort_order = st.selectbox("Order by:", options=["Natural Sequence", "Longest First", "Shortest First"])

        # Prepare chunk indices
        indices = list(range(len(chunks)))
        if search_query:
            indices = [i for i in indices if search_query.lower() in chunks[i].text.lower()]

        if sort_order == "Longest First":
            indices.sort(key=lambda i: len(chunks[i].text), reverse=True)
        elif sort_order == "Shortest First":
            indices.sort(key=lambda i: len(chunks[i].text))

        st.caption(f"Showing {len(indices)} of {total_chunks} chunks")

        for idx in indices:
            c = chunks[idx]
            color = get_color(idx)
            header_meta = f" | Path: `{c.metadata.get('header_path', '')}`" if c.metadata and "header_path" in c.metadata else ""
            with st.expander(f"Chunk #{idx} — [{c.start}:{c.end}] ({len(c.text)} chars, {count_tokens(c.text)} tokens){header_meta}", expanded=(len(indices) <= 5)):
                c_col1, c_col2 = st.columns([3, 1])
                with c_col1:
                    st.markdown("**Chunk Text**")
                    st.code(c.text, language=None, wrap_lines=True)
                with c_col2:
                    st.markdown("**Span Offsets**")
                    st.code(f"start:  {c.start}\nend:    {c.end}\nlength: {len(c.text)} chars\ntokens: {count_tokens(c.text)}\nwords:  {len(c.text.split())}")
                    if c.metadata:
                        st.markdown("**Metadata**")
                        st.json(c.metadata)

    # -----------------------------------------------------
    # Tab 3: Size Distribution & Export
    # -----------------------------------------------------
    with tab_analytics:
        st.markdown("#### Chunk Size Distribution")
        df_chunks = pd.DataFrame([
            {
                "Chunk Index": c.index,
                "Start": c.start,
                "End": c.end,
                "Characters": len(c.text),
                "Tokens (tiktoken)": count_tokens(c.text),
                "Words": len(c.text.split()),
                "Metadata": json.dumps(c.metadata) if c.metadata else "",
                "Preview": c.text[:80] + ("..." if len(c.text) > 80 else ""),
            }
            for c in chunks
        ])

        chart_metric = st.radio("Chart Metric:", ["Characters", "Tokens (tiktoken)"], horizontal=True)
        st.bar_chart(df_chunks, x="Chunk Index", y=chart_metric, color="#1976D2")

        st.markdown("#### Chunks Table")
        st.dataframe(df_chunks, width="stretch")

        st.markdown("#### Export Data")
        exp_col1, exp_col2 = st.columns(2)
        with exp_col1:
            csv_data = df_chunks.to_csv(index=False).encode("utf-8")
            st.download_button(
                label="📥 Download Chunks as CSV",
                data=csv_data,
                file_name="chunks_export.csv",
                mime="text/csv",
            )
        with exp_col2:
            export_payload = [
                {
                    "doc_id": c.doc_id,
                    "index": c.index,
                    "start": c.start,
                    "end": c.end,
                    "length": len(c.text),
                    "tokens": count_tokens(c.text),
                    "text": c.text,
                    "metadata": c.metadata,
                }
                for c in chunks
            ]
            json_data = json.dumps(export_payload, indent=2).encode("utf-8")
            st.download_button(
                label="📥 Download Chunks as JSON",
                data=json_data,
                file_name="chunks_export.json",
                mime="application/json",
            )
else:
    if not text_input.strip():
        st.info("👆 Enter some text above or choose a preset sample to view chunks.")
