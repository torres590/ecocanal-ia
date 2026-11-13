"""Componentes visuales, estilos y gráficos de ECOCANAL IA."""

from __future__ import annotations

import html
from collections.abc import Mapping
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st

from .config import (
    COLORS,
    FEATURE_COLUMNS,
    FEATURE_SHORT_LABELS,
    RISK_COLORS,
    RISK_LEVELS,
)


def inject_css() -> None:
    """Inyecta la identidad visual de la aplicación."""
    st.markdown(
        """
        <style>
        :root {
            --eco-deep: #0B3D2E;
            --eco-green: #1F8A70;
            --eco-mint: #46B98E;
            --eco-water: #1E6F9F;
            --eco-ink: #173B30;
            --eco-muted: #5C746A;
            --eco-line: #DDE7E2;
            --eco-bg: #F3F7F5;
        }
        html, body, [class*="css"], .stApp {
            font-family: "Inter", "Segoe UI", "Helvetica Neue", Arial, sans-serif;
        }
        .stApp {
            background:
                radial-gradient(circle at 92% 2%, rgba(70,185,142,.12), transparent 28rem),
                radial-gradient(circle at 3% 35%, rgba(30,111,159,.07), transparent 24rem),
                linear-gradient(180deg, #F7FAF8 0%, #F2F6F4 100%);
        }
        .block-container {
            max-width: 1240px;
            padding-top: 1.7rem;
            padding-bottom: 3rem;
        }
        h1, h2, h3, h4 { color: var(--eco-deep); letter-spacing: -0.025em; }
        p, li, label, .stMarkdown { color: var(--eco-ink); }
        a { color: var(--eco-water); }

        [data-testid="stSidebar"] {
            background: linear-gradient(175deg, #0B3D2E 0%, #072F24 52%, #061F18 100%);
            border-right: 1px solid rgba(255,255,255,.06);
        }
        [data-testid="stSidebar"] > div:first-child { padding-top: 1.1rem; }
        [data-testid="stSidebar"] * { color: #F2FAF6; }
        [data-testid="stSidebar"] hr { border-color: rgba(255,255,255,.13); }
        [data-testid="stSidebar"] [data-testid="stCaptionContainer"] p,
        [data-testid="stSidebar"] .eco-side-muted { color: #9DBBAE !important; }
        [data-testid="stSidebar"] [data-testid="stImage"] img {
            border-radius: 20px;
            box-shadow: 0 14px 35px rgba(0,0,0,.28);
        }
        [data-testid="stSidebar"] div[role="radiogroup"] { gap: .28rem; }
        [data-testid="stSidebar"] div[role="radiogroup"] label {
            padding: .55rem .75rem;
            border-radius: 11px;
            border: 1px solid transparent;
            transition: background .15s ease, border-color .15s ease, transform .15s ease;
        }
        [data-testid="stSidebar"] div[role="radiogroup"] label:hover {
            background: rgba(255,255,255,.075);
            transform: translateX(2px);
        }
        [data-testid="stSidebar"] div[role="radiogroup"] label:has(input:checked) {
            background: linear-gradient(90deg, rgba(70,185,142,.26), rgba(70,185,142,.08));
            border-color: rgba(70,185,142,.55);
        }
        [data-testid="stSidebar"] div[role="radiogroup"] label > div:first-child:not([data-testid="stMarkdownContainer"]) {
            display: none;
        }
        [data-testid="stSidebar"] div[role="radiogroup"] [data-testid="stMarkdownContainer"] p {
            font-size: .92rem;
            font-weight: 600;
        }

        .eco-hero {
            position: relative;
            overflow: hidden;
            border-radius: 28px;
            padding: 40px 44px 42px;
            background:
                radial-gradient(circle at 83% 15%, rgba(99,183,216,.34), transparent 18rem),
                radial-gradient(circle at 76% 92%, rgba(70,185,142,.30), transparent 20rem),
                linear-gradient(132deg, #0B3D2E 0%, #0D4A37 50%, #155F6E 100%);
            box-shadow: 0 26px 60px rgba(11,61,46,.25);
            margin-bottom: 1.7rem;
        }
        .eco-hero::before {
            content: "";
            position: absolute;
            width: 330px;
            height: 330px;
            border: 1px solid rgba(255,255,255,.13);
            border-radius: 50%;
            right: -80px;
            top: -120px;
        }
        .eco-hero::after {
            content: "";
            position: absolute;
            width: 230px;
            height: 230px;
            border: 1px solid rgba(255,255,255,.10);
            border-radius: 50%;
            right: 50px;
            bottom: -160px;
        }
        .eco-kicker {
            position: relative;
            z-index: 1;
            text-transform: uppercase;
            letter-spacing: .22em;
            font-size: .69rem;
            font-weight: 800;
            color: #9FE6C8;
        }
        .eco-hero h1 {
            position: relative;
            z-index: 1;
            color: #FFFFFF;
            font-size: clamp(2.3rem, 5vw, 3.55rem);
            line-height: 1.02;
            margin: .35rem 0 .5rem;
            font-weight: 850;
            letter-spacing: -0.045em;
        }
        .eco-hero .eco-slogan {
            position: relative;
            z-index: 1;
            color: #D8F5E9;
            font-size: 1.06rem;
            font-weight: 600;
            margin: 0 0 .9rem;
        }
        .eco-hero p.eco-description {
            position: relative;
            z-index: 1;
            color: #C9E2D8;
            max-width: 770px;
            line-height: 1.65;
            font-size: .95rem;
            margin: 0 0 1.15rem;
        }
        .eco-chips { position: relative; z-index: 1; display: flex; flex-wrap: wrap; gap: .5rem; }
        .eco-chip {
            display: inline-flex;
            align-items: center;
            gap: .42rem;
            padding: .34rem .7rem;
            border-radius: 999px;
            border: 1px solid rgba(255,255,255,.20);
            background: rgba(255,255,255,.095);
            color: #EFFCF7;
            font-size: .73rem;
            font-weight: 650;
            letter-spacing: .015em;
        }
        .eco-chip::before {
            content: "";
            width: 6px;
            height: 6px;
            border-radius: 50%;
            background: #7FE0C0;
        }

        .eco-page-head { margin: .1rem 0 1.45rem; }
        .eco-eyebrow {
            text-transform: uppercase;
            letter-spacing: .16em;
            font-size: .69rem;
            font-weight: 850;
            color: var(--eco-green);
            margin-bottom: .35rem;
        }
        .eco-page-head h2 {
            font-size: clamp(1.55rem, 3vw, 2.1rem);
            margin: 0 0 .35rem;
            font-weight: 820;
        }
        .eco-page-head p {
            color: var(--eco-muted);
            max-width: 850px;
            margin: 0;
            line-height: 1.6;
            font-size: .94rem;
        }

        .eco-card {
            background: #FFFFFF;
            border: 1px solid var(--eco-line);
            border-radius: 18px;
            padding: 18px 20px;
            box-shadow: 0 10px 28px rgba(11,61,46,.055);
            height: 100%;
            margin-bottom: .25rem;
        }
        .eco-metric-label {
            font-size: .69rem;
            text-transform: uppercase;
            letter-spacing: .1em;
            color: var(--eco-muted);
            font-weight: 800;
            margin-bottom: .35rem;
        }
        .eco-metric-value {
            font-size: 2.05rem;
            line-height: 1.05;
            font-weight: 860;
            color: var(--eco-deep);
            letter-spacing: -0.045em;
        }
        .eco-metric-hint { color: var(--eco-muted); font-size: .75rem; margin-top: .32rem; }
        .eco-card-title { font-weight: 800; color: var(--eco-deep); font-size: .98rem; margin-bottom: .25rem; }
        .eco-card-sub { color: var(--eco-muted); font-size: .78rem; line-height: 1.45; }

        .eco-badge {
            display: inline-flex;
            align-items: center;
            gap: .35rem;
            padding: .28rem .62rem;
            border-radius: 999px;
            font-size: .72rem;
            font-weight: 800;
            letter-spacing: .02em;
            white-space: nowrap;
        }
        .eco-badge::before {
            content: "";
            width: 7px;
            height: 7px;
            border-radius: 50%;
            background: currentColor;
        }
        .eco-status-dot {
            display: inline-block;
            width: 8px;
            height: 8px;
            border-radius: 50%;
            background: #46B98E;
            box-shadow: 0 0 0 5px rgba(70,185,142,.12);
            margin-right: .45rem;
        }

        .eco-prob-row { margin: .6rem 0 .85rem; }
        .eco-prob-head {
            display: flex;
            justify-content: space-between;
            gap: 1rem;
            margin-bottom: .3rem;
            font-size: .8rem;
            font-weight: 700;
            color: var(--eco-ink);
        }
        .eco-prob-track {
            height: 11px;
            background: #E9F0ED;
            border-radius: 999px;
            overflow: hidden;
        }
        .eco-prob-fill { height: 100%; border-radius: 999px; }

        .eco-result-banner {
            border-radius: 18px;
            padding: 20px 22px;
            border: 1px solid;
            margin: .3rem 0 1rem;
        }
        .eco-result-banner .label {
            font-size: .69rem;
            text-transform: uppercase;
            letter-spacing: .12em;
            font-weight: 850;
            opacity: .82;
        }
        .eco-result-banner .risk {
            font-size: 1.85rem;
            line-height: 1.2;
            font-weight: 860;
            letter-spacing: -.03em;
        }
        .eco-result-banner .detail { font-size: .82rem; margin-top: .25rem; }

        .eco-reco {
            background: #FFFFFF;
            border: 1px solid var(--eco-line);
            border-left: 4px solid var(--eco-green);
            border-radius: 12px;
            padding: 13px 16px;
            margin-bottom: .55rem;
            color: var(--eco-ink);
            font-size: .88rem;
            line-height: 1.55;
        }
        .eco-reco strong { color: var(--eco-deep); }

        .eco-policy {
            border-radius: 18px;
            padding: 20px 21px;
            border: 1px solid var(--eco-line);
            background: #FFFFFF;
            box-shadow: 0 9px 24px rgba(11,61,46,.05);
            height: 100%;
        }
        .eco-policy h4 { margin: .55rem 0 .35rem; font-size: 1.05rem; }
        .eco-policy p { color: var(--eco-muted); font-size: .82rem; line-height: 1.55; margin: 0; }
        .eco-policy .rule { font-weight: 850; font-size: .72rem; letter-spacing: .08em; text-transform: uppercase; }

        .eco-footer {
            margin-top: 2.6rem;
            padding: 20px 2px 0;
            border-top: 1px solid var(--eco-line);
            color: #6B8078;
            font-size: .76rem;
            line-height: 1.6;
        }
        .eco-footer strong { color: var(--eco-deep); }

        .stButton > button, .stDownloadButton > button,
        [data-testid="stFormSubmitButton"] > button {
            background: linear-gradient(135deg, #1F8A70 0%, #0B3D2E 100%);
            color: #FFFFFF !important;
            border: 0;
            border-radius: 11px;
            font-weight: 750;
            padding: .52rem 1.15rem;
            box-shadow: 0 8px 18px rgba(31,138,112,.18);
            transition: transform .14s ease, box-shadow .14s ease;
        }
        .stButton > button:hover, .stDownloadButton > button:hover,
        [data-testid="stFormSubmitButton"] > button:hover {
            transform: translateY(-1px);
            box-shadow: 0 11px 24px rgba(31,138,112,.25);
            color: #FFFFFF !important;
            border: 0;
        }
        .stButton > button:focus, .stDownloadButton > button:focus {
            color: #FFFFFF !important;
            border: 0;
            box-shadow: 0 0 0 3px rgba(70,185,142,.28);
        }

        [data-testid="stForm"] {
            background: #FFFFFF;
            border: 1px solid var(--eco-line);
            border-radius: 20px;
            padding: 1.35rem 1.45rem 1.5rem;
            box-shadow: 0 12px 30px rgba(11,61,46,.055);
        }
        .stTextInput input, .stNumberInput input, .stSelectbox div[data-baseweb="select"] > div,
        .stFileUploader section, .stTextArea textarea {
            border-radius: 10px !important;
        }
        .stNumberInput input:focus, .stTextInput input:focus {
            border-color: var(--eco-green) !important;
            box-shadow: 0 0 0 2px rgba(31,138,112,.14) !important;
        }
        [data-testid="stDataFrame"], [data-testid="stTable"] {
            border-radius: 14px;
            overflow: hidden;
            border: 1px solid var(--eco-line);
        }
        [data-testid="stExpander"] {
            border: 1px solid var(--eco-line);
            border-radius: 13px;
            background: rgba(255,255,255,.72);
            overflow: hidden;
        }
        [data-testid="stMetric"] {
            background: #FFFFFF;
            border: 1px solid var(--eco-line);
            padding: 14px 16px;
            border-radius: 15px;
            box-shadow: 0 8px 22px rgba(11,61,46,.045);
        }
        [data-testid="stAlert"] { border-radius: 12px; }
        .stTabs [data-baseweb="tab-list"] { gap: .4rem; }
        .stTabs [data-baseweb="tab"] {
            border-radius: 9px;
            padding: .45rem .95rem;
            font-weight: 700;
        }
        .stTabs [aria-selected="true"] {
            background: rgba(31,138,112,.10);
            color: var(--eco-deep);
        }
        div[data-testid="stVerticalBlockBorderWrapper"] {
            border-radius: 18px;
        }
        @media (max-width: 720px) {
            .eco-hero { padding: 30px 24px; border-radius: 22px; }
            .block-container { padding-left: 1rem; padding-right: 1rem; }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def sidebar_brand(logo_path: str, loaded: bool, dataset_rows: int | None = None) -> None:
    st.image(logo_path, width=82)
    st.markdown(
        f"""
        <div style="margin:.7rem 0 1.15rem;">
            <div style="font-size:1.48rem;font-weight:850;letter-spacing:-.04em;color:#FFFFFF;line-height:1.1;">
                ECOCANAL <span style="color:#7FE0C0;">IA</span>
            </div>
            <div style="font-size:.72rem;color:#9DBBAE;line-height:1.45;margin-top:.35rem;">
                Inteligencia artificial para proteger nuestros canales y ríos
            </div>
            <div style="margin-top:.8rem;font-size:.72rem;color:#C9E2D8;">
                <span class="eco-status-dot"></span>
                {'Modelos cargados' if loaded else 'Modelos no disponibles'}
            </div>
            {f'<div style="margin-top:.35rem;font-size:.69rem;color:#89A99B;">{dataset_rows:,} canales en el dataset sintético</div>' if dataset_rows else ''}
        </div>
        """,
        unsafe_allow_html=True,
    )


def sidebar_section(title: str) -> None:
    st.markdown(
        f'<div style="font-size:.64rem;letter-spacing:.16em;text-transform:uppercase;'
        f'color:#7EA091;font-weight:850;margin:1.1rem 0 .35rem;">{html.escape(title)}</div>',
        unsafe_allow_html=True,
    )


def hero() -> None:
    st.markdown(
        """
        <div class="eco-hero">
            <div class="eco-kicker">Observatorio predictivo de canales urbanos</div>
            <h1>ECOCANAL IA</h1>
            <p class="eco-slogan">Inteligencia artificial para proteger nuestros canales y ríos</p>
            <p class="eco-description">
                Analiza condiciones operativas de canales, estima el riesgo de acumulación
                de residuos y prioriza acciones preventivas antes de que los materiales
                lleguen a los ríos. Dos modelos de clasificación trabajan sobre las mismas
                variables para ofrecer una lectura comparada y transparente.
            </p>
            <div class="eco-chips">
                <span class="eco-chip">Regresión Logística</span>
                <span class="eco-chip">Red Neuronal Artificial</span>
                <span class="eco-chip">Dataset sintético documentado</span>
                <span class="eco-chip">Prototipo académico</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def page_header(eyebrow: str, title: str, subtitle: str) -> None:
    st.markdown(
        f"""
        <div class="eco-page-head">
            <div class="eco-eyebrow">{html.escape(eyebrow)}</div>
            <h2>{html.escape(title)}</h2>
            <p>{subtitle}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def metric_card(label: str, value: str, hint: str = "", color: str = "#1F8A70") -> None:
    st.markdown(
        f"""
        <div class="eco-card" style="border-top:4px solid {color};">
            <div class="eco-metric-label">{html.escape(label)}</div>
            <div class="eco-metric-value" style="color:{color};">{html.escape(value)}</div>
            <div class="eco-metric-hint">{html.escape(hint)}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def risk_badge(risk_level: str) -> str:
    color = RISK_COLORS.get(risk_level, "#5C746A")
    return (
        f'<span class="eco-badge" style="background:{color}18;color:{color};'
        f'border:1px solid {color}55;">{html.escape(risk_level)}</span>'
    )


def probability_bar(label: str, probability: float, risk_level: str) -> None:
    color = RISK_COLORS.get(risk_level, COLORS["verde"])
    width = max(0.0, min(100.0, probability * 100))
    st.markdown(
        f"""
        <div class="eco-prob-row">
            <div class="eco-prob-head">
                <span>{html.escape(label)}</span>
                <span style="color:{color};">{width:.1f}%</span>
            </div>
            <div class="eco-prob-track">
                <div class="eco-prob-fill" style="width:{width:.2f}%;background:{color};"></div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def result_banner(risk_level: str, probability: float, detail: str) -> None:
    color = RISK_COLORS.get(risk_level, COLORS["verde"])
    st.markdown(
        f"""
        <div class="eco-result-banner" style="border-color:{color}55;background:linear-gradient(135deg,{color}14,{color}06);">
            <div class="label" style="color:{color};">Nivel de riesgo consensuado</div>
            <div class="risk" style="color:{color};">{html.escape(risk_level)}</div>
            <div class="detail" style="color:{color};">Probabilidad promedio: {probability:.1%} · {html.escape(detail)}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def recommendation_card(text: str, index: int | None = None) -> None:
    prefix = f"<strong>{index}.</strong> " if index is not None else ""
    st.markdown(
        f'<div class="eco-reco">{prefix}{html.escape(text)}</div>',
        unsafe_allow_html=True,
    )


def policy_card(
    risk_level: str, description: str, action: str, color: str | None = None
) -> None:
    color = color or RISK_COLORS[risk_level]
    st.markdown(
        f"""
        <div class="eco-policy" style="border-top:4px solid {color};">
            <div class="rule" style="color:{color};">Riesgo {html.escape(risk_level.lower())}</div>
            <h4>{html.escape(action)}</h4>
            <p>{html.escape(description)}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def status_card(title: str, text: str, color: str = "#1F8A70") -> None:
    st.markdown(
        f"""
        <div class="eco-card" style="border-left:4px solid {color};">
            <div class="eco-card-title">{html.escape(title)}</div>
            <div class="eco-card-sub">{html.escape(text)}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def footer() -> None:
    st.markdown(
        """
        <div class="eco-footer">
            <strong>ECOCANAL IA</strong> · Prototipo académico de predicción.
            Los datos son sintéticos y las salidas son estimaciones orientativas.
            No sustituye mediciones ambientales, inspecciones técnicas ni decisiones
            de autoridades competentes.
        </div>
        """,
        unsafe_allow_html=True,
    )


def _style_figure(fig: plt.Figure) -> None:
    fig.patch.set_facecolor("none")
    for axis in fig.axes:
        axis.set_facecolor("none")


def _style_axis(axis: plt.Axes) -> None:
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    axis.spines["left"].set_color("#DDE7E2")
    axis.spines["bottom"].set_color("#DDE7E2")
    axis.tick_params(colors="#5C746A", labelsize=9)
    axis.grid(axis="y", color="#E7EFEB", linewidth=0.8, alpha=0.9)
    axis.set_axisbelow(True)


def risk_donut(counts: Mapping[str, int]) -> plt.Figure:
    values = [int(counts.get(level, 0)) for level in RISK_LEVELS]
    colors = [RISK_COLORS[level] for level in RISK_LEVELS]
    fig, ax = plt.subplots(figsize=(4.4, 3.3))
    total = sum(values)
    if total == 0:
        ax.text(0.5, 0.5, "Sin datos", ha="center", va="center")
        ax.axis("off")
    else:
        wedges, _ = ax.pie(
            values,
            colors=colors,
            startangle=90,
            counterclock=False,
            wedgeprops={"width": 0.34, "edgecolor": "#FFFFFF", "linewidth": 2.5},
        )
        ax.text(
            0,
            0.08,
            f"{total:,}",
            ha="center",
            va="center",
            fontsize=22,
            fontweight="bold",
            color=COLORS["verde_oscuro"],
        )
        ax.text(
            0,
            -0.18,
            "canales",
            ha="center",
            va="center",
            fontsize=9,
            color=COLORS["texto"],
        )
        ax.legend(
            wedges,
            [f"{level} · {value}" for level, value in zip(RISK_LEVELS, values)],
            loc="lower center",
            bbox_to_anchor=(0.5, -0.12),
            ncol=3,
            frameon=False,
            fontsize=8.5,
        )
    ax.set_aspect("equal")
    fig.tight_layout()
    _style_figure(fig)
    return fig


def probability_histogram(
    logistic_probabilities: np.ndarray | list[float],
    neural_probabilities: np.ndarray | list[float],
) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(6.2, 3.4))
    bins = np.linspace(0, 1, 24)
    ax.hist(
        np.asarray(logistic_probabilities),
        bins=bins,
        alpha=0.58,
        color=COLORS["verde"],
        label="Regresión Logística",
        edgecolor="white",
        linewidth=0.5,
    )
    ax.hist(
        np.asarray(neural_probabilities),
        bins=bins,
        alpha=0.48,
        color=COLORS["agua"],
        label="Red Neuronal",
        edgecolor="white",
        linewidth=0.5,
    )
    ax.axvline(
        0.5,
        color="#C0392B",
        linestyle="--",
        linewidth=1.4,
        label="Umbral de riesgo alto (0.50)",
    )
    ax.set_xlabel("Probabilidad estimada de riesgo alto")
    ax.set_ylabel("Cantidad de canales")
    ax.set_xlim(0, 1)
    ax.legend(frameon=False, fontsize=8.5)
    _style_axis(ax)
    fig.tight_layout()
    _style_figure(fig)
    return fig


def correlation_heatmap(dataset: pd.DataFrame) -> plt.Figure:
    columns = list(FEATURE_COLUMNS) + ["riesgo_alto"]
    labels = [FEATURE_SHORT_LABELS.get(column, column) for column in columns]
    corr = dataset[columns].corr()
    fig, ax = plt.subplots(figsize=(6.7, 4.6))
    sns.heatmap(
        corr,
        mask=np.triu(np.ones_like(corr, dtype=bool)),
        annot=True,
        fmt=".2f",
        cmap="YlGnBu",
        vmin=-1,
        vmax=1,
        square=True,
        linewidths=2,
        linecolor="#F3F7F5",
        cbar_kws={"shrink": 0.72},
        annot_kws={"fontsize": 8},
        ax=ax,
    )
    ax.set_xticklabels(labels, rotation=35, ha="right", fontsize=8)
    ax.set_yticklabels(labels, rotation=0, fontsize=8)
    _style_figure(fig)
    fig.tight_layout()
    return fig


def metrics_comparison_chart(metadata: Mapping[str, Any]) -> plt.Figure:
    metric_keys = ["accuracy", "precision", "recall", "f1"]
    metric_labels = ["Accuracy", "Precision", "Recall", "F1-score"]
    models = metadata.get("models", {})
    logistic = [float(models.get("logistica", {}).get(key, 0)) for key in metric_keys]
    neural = [float(models.get("red_neuronal", {}).get(key, 0)) for key in metric_keys]
    x = np.arange(len(metric_keys))
    width = 0.34
    fig, ax = plt.subplots(figsize=(6.8, 3.5))
    bars_log = ax.bar(
        x - width / 2,
        logistic,
        width,
        label="Regresión Logística",
        color=COLORS["verde"],
        edgecolor="white",
        linewidth=1,
    )
    bars_nn = ax.bar(
        x + width / 2,
        neural,
        width,
        label="Red Neuronal",
        color=COLORS["agua"],
        edgecolor="white",
        linewidth=1,
    )
    for bars in (bars_log, bars_nn):
        for bar in bars:
            height = bar.get_height()
            ax.annotate(
                f"{height:.2f}",
                xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, 4),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=8,
                color=COLORS["texto"],
            )
    ax.set_xticks(x, metric_labels)
    ax.set_ylim(0, 1.08)
    ax.set_ylabel("Puntuación")
    ax.legend(frameon=False, fontsize=8.5, loc="lower right")
    _style_axis(ax)
    fig.tight_layout()
    _style_figure(fig)
    return fig


def confusion_matrix_heatmap(
    matrix: list[list[int]], title: str, color: str = "Greens"
) -> plt.Figure:
    values = np.asarray(matrix, dtype=int)
    fig, ax = plt.subplots(figsize=(3.9, 3.1))
    sns.heatmap(
        values,
        annot=True,
        fmt="d",
        cmap=color,
        cbar=False,
        linewidths=3,
        linecolor="#FFFFFF",
        annot_kws={"fontsize": 13, "fontweight": "bold"},
        ax=ax,
    )
    ax.set_xlabel("Predicción", fontsize=9)
    ax.set_ylabel("Etiqueta real", fontsize=9)
    ax.set_xticklabels(["Sin alto", "Alto"], fontsize=8)
    ax.set_yticklabels(["Sin alto", "Alto"], fontsize=8, rotation=0)
    ax.set_title(title, fontsize=10, color=COLORS["verde_oscuro"], pad=10, fontweight="bold")
    _style_figure(fig)
    fig.tight_layout()
    return fig


def training_history_chart(metadata: Mapping[str, Any]) -> plt.Figure:
    history = metadata.get("models", {}).get("red_neuronal", {}).get("history", {})
    fig, ax = plt.subplots(figsize=(6.4, 3.2))
    if history:
        epochs = np.arange(1, len(history.get("loss", [])) + 1)
        ax.plot(epochs, history.get("loss", []), color=COLORS["agua"], linewidth=2, label="Pérdida entrenamiento")
        ax.plot(
            epochs,
            history.get("val_loss", []),
            color=COLORS["verde"],
            linewidth=2,
            label="Pérdida validación interna",
        )
        ax.set_xlabel("Época")
        ax.set_ylabel("Binary cross-entropy")
        ax.legend(frameon=False, fontsize=8.5)
        _style_axis(ax)
    else:
        ax.text(0.5, 0.5, "Historial no disponible", ha="center", va="center")
        ax.axis("off")
    fig.tight_layout()
    _style_figure(fig)
    return fig


def logistic_coefficients_chart(metadata: Mapping[str, Any]) -> plt.Figure:
    coefficients = (
        metadata.get("models", {}).get("logistica", {}).get("coefficients", {})
    )
    fig, ax = plt.subplots(figsize=(6.4, 3.5))
    if coefficients:
        items = sorted(
            coefficients.items(), key=lambda item: item[1]["coeficiente"]
        )
        labels = [FEATURE_SHORT_LABELS.get(key, key) for key, _ in items]
        values = [item[1]["coeficiente"] for item in items]
        colors = [
            COLORS["verde"] if value >= 0 else "#C0392B" for value in values
        ]
        ax.barh(labels, values, color=colors, alpha=0.88)
        ax.axvline(0, color="#5C746A", linewidth=1)
        ax.set_xlabel("Coeficiente estandarizado")
        _style_axis(ax)
        ax.grid(axis="y", visible=False)
    else:
        ax.text(0.5, 0.5, "Coeficientes no disponibles", ha="center", va="center")
        ax.axis("off")
    fig.tight_layout()
    _style_figure(fig)
    return fig


def priority_bar(level: str, count: int, total: int) -> None:
    color = RISK_COLORS[level]
    pct = (count / total * 100) if total else 0
    st.markdown(
        f"""
        <div style="margin:.35rem 0 .8rem;">
            <div style="display:flex;justify-content:space-between;font-size:.78rem;font-weight:700;margin-bottom:.25rem;">
                <span>{html.escape(level)}</span><span style="color:{color};">{count:,} · {pct:.1f}%</span>
            </div>
            <div style="height:10px;background:#E9F0ED;border-radius:999px;overflow:hidden;">
                <div style="height:100%;width:{pct:.2f}%;background:{color};border-radius:999px;"></div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
