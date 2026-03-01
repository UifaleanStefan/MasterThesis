"""Figure 1 — Memory Graph: Fixed theta vs. Learned theta (side-by-side)."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import networkx as nx
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

if TYPE_CHECKING:
    pass


def _node_colors_and_shapes(graph: nx.DiGraph) -> tuple[list[str], list[str]]:
    colors = []
    labels = {}
    event_nodes = []
    entity_nodes = []
    for node, data in graph.nodes(data=True):
        if data.get("type") == "event":
            colors.append("#4C9BE8")  # blue
            labels[node] = f"e{data.get('step', '?')}"
            event_nodes.append(node)
        else:
            colors.append("#F4A261")  # orange
            labels[node] = str(node).replace("_", "\n")
            entity_nodes.append(node)
    return colors, labels, event_nodes, entity_nodes


def _edge_colors_and_styles(graph: nx.DiGraph):
    temporal_edges = []
    mention_edges = []
    for u, v, data in graph.edges(data=True):
        etype = data.get("edge_type", "")
        if etype == "temporal":
            temporal_edges.append((u, v))
        else:
            mention_edges.append((u, v))
    return temporal_edges, mention_edges


def _draw_graph(ax: plt.Axes, graph: nx.DiGraph, title: str) -> None:
    if graph.number_of_nodes() == 0:
        ax.text(0.5, 0.5, "Empty graph", ha="center", va="center", transform=ax.transAxes)
        ax.set_title(title, fontsize=12, fontweight="bold")
        ax.axis("off")
        return

    pos = nx.spring_layout(graph, seed=42, k=1.5)
    colors, labels, event_nodes, entity_nodes = _node_colors_and_shapes(graph)
    temporal_edges, mention_edges = _edge_colors_and_styles(graph)

    node_color_map = []
    for node in graph.nodes():
        data = graph.nodes[node]
        node_color_map.append("#4C9BE8" if data.get("type") == "event" else "#F4A261")

    nx.draw_networkx_nodes(graph, pos, ax=ax, node_color=node_color_map,
                           node_size=300, alpha=0.9)
    nx.draw_networkx_labels(graph, pos, labels=labels, ax=ax, font_size=6)

    if temporal_edges:
        nx.draw_networkx_edges(graph, pos, edgelist=temporal_edges, ax=ax,
                               edge_color="#888888", arrows=True,
                               arrowsize=12, width=1.2,
                               connectionstyle="arc3,rad=0.05")
    if mention_edges:
        nx.draw_networkx_edges(graph, pos, edgelist=mention_edges, ax=ax,
                               edge_color="#E76F51", arrows=True,
                               style="dashed", arrowsize=10, width=0.8,
                               connectionstyle="arc3,rad=0.1")

    stats = {k: v for k, v in [
        ("nodes", graph.number_of_nodes()),
        ("edges", graph.number_of_edges()),
        ("events", sum(1 for _, d in graph.nodes(data=True) if d.get("type") == "event")),
        ("entities", sum(1 for _, d in graph.nodes(data=True) if d.get("type") == "entity")),
    ]}
    stat_str = f"nodes={stats['nodes']}  edges={stats['edges']}\nevents={stats['events']}  entities={stats['entities']}"
    ax.text(0.02, 0.02, stat_str, transform=ax.transAxes, fontsize=7,
            verticalalignment="bottom",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.7))

    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.axis("off")


def plot_memory_graphs(
    fixed_graph: nx.DiGraph,
    learned_graph: nx.DiGraph,
    fixed_theta: tuple,
    learned_theta: tuple,
    env_name: str,
    output_dir: str | Path,
) -> Path:
    """Render side-by-side fixed vs. learned theta memory graph for one episode."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(14, 7))
    fig.suptitle(
        f"Memory Graph — {env_name}  (same episode, seed=42)",
        fontsize=14, fontweight="bold", y=1.01
    )

    ts, te, tt = fixed_theta
    _draw_graph(axes[0], fixed_graph,
                f"Fixed theta\n(store={ts:.2f}, entity={te:.2f}, temporal={tt:.2f})")

    ts, te, tt = learned_theta
    _draw_graph(axes[1], learned_graph,
                f"Learned theta (ES)\n(store={ts:.2f}, entity={te:.2f}, temporal={tt:.2f})")

    # Legend
    legend_elements = [
        mpatches.Patch(facecolor="#4C9BE8", label="Event node"),
        mpatches.Patch(facecolor="#F4A261", label="Entity node"),
        mpatches.Patch(facecolor="#888888", label="Temporal edge"),
        mpatches.Patch(facecolor="#E76F51", label="Mention edge"),
    ]
    fig.legend(handles=legend_elements, loc="lower center", ncol=4,
               bbox_to_anchor=(0.5, -0.04), fontsize=9)

    plt.tight_layout()
    out_path = output_dir / f"fig1_memory_graphs_{env_name.lower().replace('-', '_').replace(' ', '_')}.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out_path
