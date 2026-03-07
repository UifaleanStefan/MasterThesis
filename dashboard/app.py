"""
Learnable Memory Thesis — Interactive Dashboard

Story-led exploration: The Question → Approach → Evidence (Benchmark, Ablation, Sensitivity)
→ Task-dependent Transfer → Neural meta-controller → Compare & explore → Figures → Playground.

Usage (PowerShell, from project root):
  streamlit run dashboard/app.py

Requires: results/*.json (run experiments first; see docs/RUNNING_EXPERIMENTS.md).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
RESULTS_DIR = ROOT / "results"
FIGS_DIR = ROOT / "docs" / "figures"

import streamlit as st
import pandas as pd

from dashboard import copy as copy_module
from dashboard.charts import (
    bar_reward_by_system,
    scatter_precision_reward,
    heatmap_sensitivity,
    line_training_curve,
    bar_degradation,
)


@st.cache_data
def load_json(name: str) -> dict | None:
    p = RESULTS_DIR / name
    if not p.exists():
        return None
    with open(p) as f:
        return json.load(f)


def render_section_title(title: str, narrative: str) -> None:
    st.markdown(f"### {title}")
    st.markdown(narrative)
    st.markdown("")


def render_callout(label: str, value: str | float, delta: str | None = None) -> None:
    st.metric(label, value, delta=delta)


# --- Story navigation (plan order) ---
SECTIONS = [
    "The Question",
    "The Approach",
    "Evidence: Benchmark",
    "Evidence: Ablation",
    "Evidence: Sensitivity",
    "Task-dependent memory",
    "Neural meta-controller",
    "Compare & explore",
    "All figures",
    "Playground",
    "What's next",
]

st.set_page_config(page_title="Learnable Memory Thesis", layout="wide")

# Sidebar
st.sidebar.markdown("### Navigation")
story_mode = st.sidebar.checkbox("Story mode (single scroll)", value=False)
if story_mode:
    st.sidebar.caption("All sections in one page. Use the page to scroll.")
    section = None
else:
    section = st.sidebar.radio("Section", SECTIONS, label_visibility="collapsed")

st.sidebar.markdown("---")
st.sidebar.caption("Data: results/*.json. See docs/RUNNING_EXPERIMENTS.md")

# Hero (always at top when not story mode)
if not story_mode and section:
    st.title("Learnable Memory Construction — Thesis Dashboard")
    st.markdown(
        "**Research question:** Can an agent learn how to construct its own memory representation, "
        "and should that structure be task-dependent?"
    )
    st.markdown("---")


def render_question() -> None:
    render_section_title("The Question", copy_module.QUESTION_HOOK)
    st.markdown(copy_module.QUESTION_CLAIM)
    st.markdown("")
    st.info(copy_module.QUESTION_ONE_LINE)


def render_approach() -> None:
    render_section_title("The Approach", copy_module.APPROACH_INTRO)
    for b in copy_module.APPROACH_BULLETS:
        st.markdown(f"- {b}")
    st.markdown("Go to **Evidence: Benchmark** and **Compare & explore** for results.")


def _benchmark_df_and_envs() -> tuple[pd.DataFrame | None, list[str], list[str]]:
    data = load_json("benchmark_results.json")
    if not data:
        return None, [], []
    envs = list(data.keys())
    systems = list(next(iter(data.values())).keys()) if data else []
    rows = []
    for sys_name in systems:
        row = {"System": sys_name}
        for env_name in envs:
            res = data.get(env_name, {}).get(sys_name, {})
            if "error" in res:
                row[f"{env_name}_reward"] = None
                row[f"{env_name}_precision"] = None
            else:
                row[f"{env_name}_reward"] = res.get("mean_reward")
                row[f"{env_name}_precision"] = res.get("retrieval_precision")
        rows.append(row)
    df = pd.DataFrame(rows)
    return df, envs, systems


def render_evidence_benchmark() -> None:
    render_section_title("Evidence: Benchmark", copy_module.EVIDENCE_BENCHMARK)
    data = load_json("benchmark_results.json")
    if data is None:
        st.warning(copy_module.empty_state("benchmark", copy_module.COMMANDS["benchmark"]))
        st.code(copy_module.COMMANDS["benchmark"], language="text")
        return
    df, envs, systems = _benchmark_df_and_envs()
    if df is None or not len(df):
        return
    # Key callout: V4 on MultiHop if present
    multihop = "MultiHopKeyDoor" if "MultiHopKeyDoor" in envs else (envs[0] if envs else None)
    if multihop:
        v4_row = df[df["System"] == "GraphMemoryV4"]
        if not v4_row.empty:
            r = v4_row[f"{multihop}_reward"].iloc[0]
            p = v4_row[f"{multihop}_precision"].iloc[0]
            col1, col2, col3 = st.columns(3)
            with col1:
                render_callout("GraphMemoryV4 mean reward (MultiHop)", f"{r:.3f}" if r is not None else "N/A")
            with col2:
                render_callout("Retrieval precision", f"{p:.3f}" if p is not None else "N/A")
            with col3:
                st.markdown(f"**Takeaway:** {copy_module.BENCHMARK_KEY_TAKEAWAY}")
    st.markdown("")
    # Filters
    env = st.selectbox("Environment", envs, key="bench_env")
    reward_col = f"{env}_reward"
    precision_col = f"{env}_precision"
    default_systems = [
        "GraphMemoryV4",
        "GraphMemoryV1",
        "EpisodicSemantic",
        "WorkingMemory(7)",
        "SemanticMemory",
        "FlatWindow(50)",
    ]
    available = [s for s in default_systems if s in systems]
    selected = st.multiselect(
        "Systems (select to compare)",
        systems,
        default=available if available else systems[:6],
        key="bench_systems",
    )
    if not selected:
        selected = systems[:8]
    df_sel = df[df["System"].isin(selected)].copy()
    df_sel["reward"] = df_sel[reward_col]
    df_sel["precision"] = df_sel[precision_col]
    df_sel = df_sel.dropna(subset=["reward"])
    # Bar chart
    tab1, tab2, tab3 = st.tabs(["Interactive chart", "Precision–reward scatter", "Table"])
    with tab1:
        df_bar = df_sel[["System", "reward"]].rename(columns={"reward": env})
        fig = bar_reward_by_system(df_bar, env, title=f"Mean reward by system — {env}")
        if fig:
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.bar_chart(df_sel.set_index("System")["reward"])
    with tab2:
        df_scatter = df_sel[["System", "precision", "reward"]].dropna()
        if not df_scatter.empty:
            fig2 = scatter_precision_reward(
                df_scatter,
                precision_col="precision",
                reward_col="reward",
                title=f"Precision vs reward — {env}",
            )
            if fig2:
                st.plotly_chart(fig2, use_container_width=True)
            else:
                st.scatter_chart(df_scatter, x="precision", y="reward")
        else:
            st.caption("Precision not available for this environment.")
    with tab3:
        show_df = df_sel[["System", "reward", "precision"]].rename(columns={"reward": f"reward ({env})", "precision": "precision"})
        st.dataframe(show_df, use_container_width=True)
    st.caption("Optional: merge V4/V1 from results/graphmemory_v4_cmaes_results.json for CMA-ES runs.")


def render_evidence_ablation() -> None:
    render_section_title("Evidence: Ablation", copy_module.EVIDENCE_ABLATION)
    data = load_json("ablation_results.json")
    if data is None:
        st.warning(copy_module.empty_state("ablation", copy_module.COMMANDS["ablation"]))
        st.code(copy_module.COMMANDS["ablation"], language="text")
        return
    results = data.get("results", data)
    if not results:
        return
    configs = []
    degradation_pct = []
    mean_reward_list = []
    for name, res in results.items():
        configs.append(name)
        deg = res.get("degradation")
        degradation_pct.append((deg * 100) if deg is not None else 0)
        mean_reward_list.append(res.get("mean_reward"))
    # Callout
    no_novel = next((d for c, d in zip(configs, degradation_pct) if "novel" in c.lower()), None)
    if no_novel is not None:
        st.info(f"**{copy_module.ABLATION_KEY_TAKEAWAY}** (degradation: {no_novel:.0f}%)")
    # Sort by degradation desc
    order = sorted(range(len(configs)), key=lambda i: -degradation_pct[i])
    configs = [configs[i] for i in order]
    degradation_pct = [degradation_pct[i] for i in order]
    tab1, tab2 = st.tabs(["Interactive chart", "Table"])
    with tab1:
        fig = bar_degradation(configs, degradation_pct)
        if fig:
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.bar_chart(pd.DataFrame({"Config": configs, "Degradation %": degradation_pct}).set_index("Config"))
    with tab2:
        rows = [{"Config": c, "Degradation %": d, "Mean reward": results.get(c, {}).get("mean_reward")} for c, d in zip(configs, degradation_pct)]
        st.dataframe(pd.DataFrame(rows), use_container_width=True)


def render_evidence_sensitivity() -> None:
    render_section_title("Evidence: Sensitivity", copy_module.EVIDENCE_SENSITIVITY)
    data = load_json("sensitivity_results.json")
    if data is None:
        st.warning(copy_module.empty_state("sensitivity", copy_module.COMMANDS["sensitivity"]))
        st.code(copy_module.COMMANDS["sensitivity"], language="text")
        return
    dim1 = data.get("dim1_values", [])
    dim2 = data.get("dim2_values", [])
    grid = data.get("reward_grid", [])
    learned1 = data.get("learned_dim1")
    learned2 = data.get("learned_dim2")
    best = data.get("best_reward")
    st.info(f"**{copy_module.SENSITIVITY_KEY_TAKEAWAY}**")
    col1, col2, col3 = st.columns(3)
    with col1:
        render_callout("Best reward", f"{best:.3f}" if best is not None else "N/A")
    with col2:
        render_callout("Learned theta_novel", f"{learned1:.3f}" if learned1 is not None else "N/A")
    with col3:
        render_callout("Learned w_recency", f"{learned2:.3f}" if learned2 is not None else "N/A")
    cfg = data.get("config", {})
    dim1_name = cfg.get("dim1", "theta_novel")
    dim2_name = cfg.get("dim2", "w_recency")
    tab1, tab2 = st.tabs(["Interactive heatmap", "Publication figure"])
    with tab1:
        fig = heatmap_sensitivity(dim1, dim2, grid, learned1, learned2, dim1_name, dim2_name)
        if fig:
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.caption("Install plotly for interactive heatmap. See fig_sensitivity_annotated.png below.")
    with tab2:
        p = FIGS_DIR / "fig_sensitivity_annotated.png"
        if p.exists():
            st.image(str(p), use_container_width=True)
        else:
            st.caption("Run generate_thesis_figures.py to create fig_sensitivity_annotated.png.")


def render_transfer() -> None:
    render_section_title("Task-dependent memory", copy_module.TRANSFER_NARRATIVE)
    data = load_json("transfer_results.json")
    if data is None:
        st.warning(copy_module.empty_state("transfer", copy_module.COMMANDS["transfer"]))
        st.code(copy_module.COMMANDS["transfer"], language="text")
        return
    matrix = data.get("matrix", {})
    zs_key = "MultiHop_V4_zeroshot"
    if zs_key not in matrix:
        st.json(matrix)
        return
    zs = matrix[zs_key]
    envs = list(zs.keys())
    st.info(f"**{copy_module.TRANSFER_KEY_TAKEAWAY}**")
    target = st.selectbox("Target environment", envs, key="transfer_env")
    rec = zs.get(target, {})
    r = rec.get("mean_reward")
    t = rec.get("mean_tokens")
    col1, col2, col3 = st.columns(3)
    with col1:
        render_callout(f"V4 zero-shot reward ({target})", f"{r:.3f}" if r is not None else "N/A")
    with col2:
        render_callout("Mean tokens", f"{t:.0f}" if t is not None else "N/A")
    with col3:
        mh = zs.get("MultiHopKeyDoor", {}).get("mean_reward")
        ratio = (r / mh * 100) if (mh and r is not None and mh > 0) else None
        st.metric("Transfer strength (vs MultiHop)", f"{ratio:.0f}%" if ratio is not None else "N/A")
    rows = [{"Environment": e, "Mean reward": zs[e]["mean_reward"], "Mean tokens": zs[e]["mean_tokens"]} for e in envs]
    st.dataframe(pd.DataFrame(rows), use_container_width=True)


def render_neural() -> None:
    render_section_title("Neural meta-controller", copy_module.NEURAL_NARRATIVE)
    data = load_json("neural_controller_v2_results.json")
    if data is None:
        st.warning(copy_module.empty_state("neural", copy_module.COMMANDS["neural"]))
        st.code(copy_module.COMMANDS["neural"], language="text")
        return
    st.info(f"**{copy_module.NEURAL_KEY_TAKEAWAY}**")
    cfg = data.get("config", {})
    tr = data.get("training", {})
    history = tr.get("history", [])
    v4_reward = None
    v4_comp = data.get("v4_scalar_comparison", {})
    if v4_comp:
        v4_reward = v4_comp.get("mean_reward")
    gens = [h["generation"] for h in history]
    best_fit = [h["best_fitness"] for h in history]
    sigma_list = [h.get("sigma") for h in history if "sigma" in h]
    if not sigma_list:
        sigma_list = None
    col1, col2, col3 = st.columns(3)
    with col1:
        render_callout("Best fitness", f"{tr.get('best_fitness', 0):.3f}")
    with col2:
        render_callout("V4 scalar reward", f"{v4_reward:.3f}" if v4_reward is not None else "N/A")
    with col3:
        delta = (tr.get("best_fitness") or 0) - (v4_reward or 0)
        render_callout("Neural vs V4", f"{delta:+.3f}", delta=None)
    tab1, tab2, tab3 = st.tabs(["Training curve", "Comparison", "Table"])
    with tab1:
        fig = line_training_curve(gens, best_fit, sigma_list, v4_reward)
        if fig:
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.line_chart(pd.DataFrame({"generation": gens, "best_fitness": best_fit}).set_index("generation"))
    with tab2:
        mh = data.get("eval_multihop") or {}
        mq = data.get("eval_megaquest") or {}
        st.metric("Neural MultiHop reward", mh.get("mean_reward"))
        st.metric("Neural MegaQuest reward", mq.get("mean_reward"))
        st.metric("V4 scalar reward", v4_comp.get("mean_reward"))
    with tab3:
        st.json({"config": cfg, "training": {"best_fitness": tr.get("best_fitness"), "elapsed_s": tr.get("elapsed_s")}})


def render_compare() -> None:
    st.markdown("### Compare & explore")
    st.markdown(copy_module.COMPARE_INTRO)
    data = load_json("benchmark_results.json")
    if data is None:
        st.warning(copy_module.empty_state("benchmark", copy_module.COMMANDS["benchmark"]))
        st.code(copy_module.COMMANDS["benchmark"], language="text")
        return
    df, envs, systems = _benchmark_df_and_envs()
    if df is None:
        return
    env = st.selectbox("Environment", envs, key="compare_env")
    reward_col = f"{env}_reward"
    precision_col = f"{env}_precision"
    selected = st.multiselect("Systems", systems, default=systems[:10], key="compare_systems")
    if not selected:
        selected = systems
    df_sel = df[df["System"].isin(selected)].copy()
    df_sel = df_sel.sort_values(reward_col, ascending=False)
    sort_by = st.selectbox("Sort table by", ["reward", "precision"], format_func=lambda x: "Mean reward" if x == "reward" else "Precision", key="compare_sort")
    col = reward_col if sort_by == "reward" else precision_col
    if col in df_sel.columns:
        df_sel = df_sel.sort_values(col, ascending=False)
    display = df_sel[["System", reward_col, precision_col]].rename(columns={reward_col: "Mean reward", precision_col: "Precision"})
    st.dataframe(display, use_container_width=True)
    df_sel["reward"] = df_sel[reward_col]
    df_sel["precision"] = df_sel[precision_col]
    fig = bar_reward_by_system(df_sel[["System", "reward"]].rename(columns={"reward": env}), env)
    if fig:
        st.plotly_chart(fig, use_container_width=True)


# Same VISUALIZATIONS list, grouped
FIG_GROUPS = {
    "Core story": [
        ("fig_master_benchmark.png", "Master Benchmark Summary (2×2)", "Four-panel overview: MultiHop ranking, CMA-ES curve, precision, efficiency.", "GraphMemoryV4 leads; precision gates reward."),
        ("fig_ablation_ranked.png", "Ablation — Component Importance", "Bar chart of degradation when each θ component is removed.", "theta_novel removal → 100% degradation."),
        ("fig_transfer_annotated.png", "Transfer Heatmap", "V4 theta across GoalRoom, HardKeyDoor, MegaQuestRoom.", "Memory is task-dependent; MegaQuest fails."),
        ("fig_sensitivity_annotated.png", "Sensitivity — Reward Landscape", "2D heatmap theta_novel × w_recency.", "Broad plateau → robust θ."),
        ("fig_neural_analysis.png", "Neural Controller Analysis", "Training curve, MultiHop comparison, transfer.", "Neural matches V4 with budget; transfer fails."),
        ("fig_neural_v2_curves.png", "Neural V2 — Learning Curve", "Best fitness and sigma vs generation.", "From neural_controller_v2_results.json."),
        ("fig_neural_transfer.png", "Neural vs V4 — Transfer", "Grouped bars: MultiHop and MegaQuest.", "Both fail on MegaQuest (OOD)."),
    ],
    "Benchmark detail": [
        ("fig5_memory_comparison.png", "Fig 5 — MultiHop Memory Comparison", "Three-panel: reward, precision, efficiency.", "Primary benchmark viz."),
        ("fig5b_cross_env_heatmap.png", "Fig 5b — Cross-Environment Heatmap", "Systems × envs heatmap.", "No single winner everywhere."),
        ("fig5c_precision_vs_reward.png", "Fig 5c — Precision vs Reward", "Scatter across (system, env).", "Precision–reward correlation."),
        ("fig5d_simple_env_comparison.png", "Fig 5d — Simple Env Comparison", "Bars across Key-Door, Goal-Room, MultiHop.", None),
        ("fig_story_precision_reward.png", "Precision vs Reward by Environment", "Scatter per environment.", None),
        ("fig11_pareto.png", "Pareto — Reward vs Token Cost", "Pareto front on MultiHop.", "V4: high reward, low tokens."),
        ("fig13_memory_size.png", "Memory Size Comparison", "Mean events stored per episode.", "V4 stores ~10 events (selective)."),
    ],
    "Extended": [
        ("fig13_memory_curves.png", "Fig 13 — Memory Over Episode", "Illustrative growth over steps.", None),
        ("fig08_ablation.png", "Fig 8 — Ablation (generic)", "Generic 3D θ ablation.", None),
        ("fig08_ablation_v4.png", "Fig 8 V4 — Ablation 10D", "V4 ablation from run_ablation.py.", None),
        ("fig09_landscape.png", "Fig 9 — Landscape (theta_store × entity)", "3D θ sensitivity.", None),
        ("fig09_landscape_v4.png", "Fig 9 V4 — theta_novel × w_recency", "From run_sensitivity.py.", None),
        ("fig10_transfer.png", "Fig 10 — Transfer Matrix", "Source × target heatmap.", None),
        ("fig10_transfer_v4.png", "Fig 10 V4 — Zero-Shot Transfer", "V4 transfer.", None),
        ("fig12_online_adaptation.png", "Fig 12 — Online Adaptation", "Placeholder θ curves.", None),
        ("fig14_cost_breakdown.png", "Fig 14 — LLM Cost Breakdown", "Placeholder cost components.", None),
        ("fig15_multi_session.png", "Fig 15 — Multi-Session", "Placeholder multi-session.", None),
    ],
}


def render_figures() -> None:
    st.markdown("### All figures")
    st.markdown("Pre-generated figures grouped by story and type. Key takeaway under each where applicable.")
    if not FIGS_DIR.exists():
        st.warning("docs/figures/ not found. Run `python generate_thesis_figures.py --allow-missing` or `python regen_all_figures.py`.")
        return
    for group_name, items in FIG_GROUPS.items():
        with st.expander(f"**{group_name}**", expanded=(group_name == "Core story")):
            for filename, title, explanation, takeaway in items:
                p = FIGS_DIR / filename
                st.markdown(f"**{title}** — *{filename}*")
                if not p.exists():
                    st.caption("(not generated yet)")
                    st.info(explanation or "")
                    continue
                try:
                    st.image(str(p), use_container_width=True)
                except Exception as e:
                    st.error(str(e))
                st.caption(explanation or "")
                if takeaway:
                    st.markdown(f"*Key takeaway:* {takeaway}")
                st.markdown("")


def render_playground() -> None:
    st.markdown("### Playground — Run one episode")
    st.caption("See how different memory systems behave in one episode.")
    env_name = st.selectbox("Environment", ["Key-Door", "Goal-Room", "MultiHop-KeyDoor", "MegaQuestRoom"], key="pg_env")
    mem_name = st.selectbox(
        "Memory system",
        ["FlatMemory", "GraphMemoryV4", "EpisodicSemantic", "SemanticMemory", "WorkingMemory(7)"],
        key="pg_mem",
    )
    k = st.slider("Retrieval k", 4, 16, 8, key="pg_k")
    if st.button("Run 1 episode"):
        try:
            from environment import ToyEnvironment, GoalRoom, MultiHopKeyDoor
            from environment.mega_quest import MegaQuestRoom
            from agent import ExplorationPolicy
            from agent.loop import run_episode_with_any_memory
            from memory.flat_memory import FlatMemory
            from memory.episodic_semantic_memory import EpisodicSemanticMemory
            from memory.semantic_memory import SemanticMemory
            from memory.working_memory import WorkingMemory
            from memory.graph_memory_v4 import GraphMemoryV4, MemoryParamsV4
            env_map = {
                "Key-Door": lambda: ToyEnvironment(seed=0),
                "Goal-Room": lambda: GoalRoom(seed=0),
                "MultiHop-KeyDoor": lambda: MultiHopKeyDoor(seed=0),
                "MegaQuestRoom": lambda: MegaQuestRoom(seed=0),
            }
            default_v4 = MemoryParamsV4(
                theta_store=0.293, theta_novel=0.908, theta_erich=0.198, theta_surprise=0.785,
                theta_entity=0.285, theta_temporal=0.278, theta_decay=0.668,
                w_graph=0.0, w_embed=1.079, w_recency=3.777, mode="learnable",
            )
            mem_map = {
                "FlatMemory": lambda: FlatMemory(window_size=50),
                "GraphMemoryV4": lambda: GraphMemoryV4(default_v4),
                "EpisodicSemantic": lambda: EpisodicSemanticMemory(episodic_size=30),
                "SemanticMemory": lambda: SemanticMemory(max_capacity=80),
                "WorkingMemory(7)": lambda: WorkingMemory(capacity=7),
            }
            env = env_map[env_name]()
            policy = ExplorationPolicy(seed=0)
            mem = mem_map[mem_name]()
            success, events, stats = run_episode_with_any_memory(env, policy, mem, k=k, episode_seed=42)
            st.success(f"Success: {success}")
            st.metric("Reward", stats.get("reward", 0))
            st.metric("Memory size", stats.get("memory_size", 0))
            st.metric("Retrieval tokens", stats.get("retrieval_tokens", 0))
            prec = stats.get("retrieval_precision")
            if prec is not None:
                st.metric("Retrieval precision", f"{prec:.4f}")
        except Exception as e:
            st.exception(e)


def render_whats_next() -> None:
    st.markdown("### What's next")
    st.markdown(copy_module.WHATS_NEXT)
    st.markdown("**Takeaways:**")
    for t in copy_module.TAKEAWAYS:
        st.markdown(f"- {t}")
    doc_qa = load_json("document_qa_memory_results.json")
    if doc_qa:
        st.markdown("**DocumentQA memory (recall@k):**")
        import pandas as pd
        rows = [{"System": s, "Mean recall": d["mean_recall"], "Std recall": d.get("std_recall")} for s, d in doc_qa.items()]
        st.dataframe(pd.DataFrame(rows).sort_values("Mean recall", ascending=False), use_container_width=True)
    st.caption("See docs/RUNNING_EXPERIMENTS.md for DocumentQA+LLM commands.")


# --- Dispatch ---
def render_all_story() -> None:
    st.title("Learnable Memory Construction — Thesis Dashboard")
    st.markdown("**Research question:** Can an agent learn how to construct its own memory representation, and should that structure be task-dependent?")
    st.markdown("---")
    render_question()
    st.markdown("---")
    render_approach()
    st.markdown("---")
    render_evidence_benchmark()
    st.markdown("---")
    render_evidence_ablation()
    st.markdown("---")
    render_evidence_sensitivity()
    st.markdown("---")
    render_transfer()
    st.markdown("---")
    render_neural()
    st.markdown("---")
    render_compare()
    st.markdown("---")
    render_figures()
    st.markdown("---")
    render_playground()
    st.markdown("---")
    render_whats_next()


SECTION_RENDERERS = {
    "The Question": render_question,
    "The Approach": render_approach,
    "Evidence: Benchmark": render_evidence_benchmark,
    "Evidence: Ablation": render_evidence_ablation,
    "Evidence: Sensitivity": render_evidence_sensitivity,
    "Task-dependent memory": render_transfer,
    "Neural meta-controller": render_neural,
    "Compare & explore": render_compare,
    "All figures": render_figures,
    "Playground": render_playground,
    "What's next": render_whats_next,
}

if story_mode:
    render_all_story()
else:
    if section in SECTION_RENDERERS:
        SECTION_RENDERERS[section]()
    else:
        render_question()
