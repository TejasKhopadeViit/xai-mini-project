"""
run_experiment.py
=================
Classification and explanation pipeline for the AIFB RDF dataset.

We follow Strategy 2a from the mini-project description:
  1. Convert RDF graph data into tabular features (1-hop neighborhood)
  2. Train interpretable classifiers (Logistic Regression, Decision Tree)
  3. Generate local explanations via model coefficients and SHAP-like perturbation
  4. Evaluate explanations through faithfulness (feature deletion) tests
  5. Produce plots and JSON outputs for the report

Usage:
    python run_experiment.py --data-dir ./data --output-dir ./results
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import warnings
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import matplotlib
matplotlib.use("Agg")  # non-interactive backend for saving plots
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import seaborn as sns

from rdflib import Graph, Literal, URIRef
from rdflib.namespace import RDF, RDFS, OWL
from sklearn.dummy import DummyClassifier
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.tree import DecisionTreeClassifier, export_text

warnings.filterwarnings("ignore", category=FutureWarning)

# ---------------------------------------------------------------------------
# Mapping from AIFB research-group URIs to short human-readable names.
# These come from the :name literals in the .n3 file.
# ---------------------------------------------------------------------------
RESEARCH_GROUP_LABELS = {
    "http://www.aifb.uni-karlsruhe.de/Forschungsgruppen/viewForschungsgruppeOWL/id1instance":
        "Business Information Systems",
    "http://www.aifb.uni-karlsruhe.de/Forschungsgruppen/viewForschungsgruppeOWL/id2instance":
        "Efficient Algorithms",
    "http://www.aifb.uni-karlsruhe.de/Forschungsgruppen/viewForschungsgruppeOWL/id3instance":
        "Knowledge Management",
    "http://www.aifb.uni-karlsruhe.de/Forschungsgruppen/viewForschungsgruppeOWL/id4instance":
        "Complexity Management",
    "http://www.aifb.uni-karlsruhe.de/Forschungsgruppen/viewForschungsgruppeOWL/id5instance":
        "Usability Engineering",
}


def readable_label(uri: str) -> str:
    """Return human-friendly label for a URI, falling back to local name."""
    if uri in RESEARCH_GROUP_LABELS:
        return RESEARCH_GROUP_LABELS[uri]
    return short_name(uri)


# ---------------------------------------------------------------------------
# RDF helpers
# ---------------------------------------------------------------------------

def short_name(value: object) -> str:
    """Extract the local part of a URI/literal for use as feature token."""
    text = str(value)
    if "#" in text:
        text = text.rsplit("#", 1)[-1]
    if "/" in text:
        text = text.rsplit("/", 1)[-1]
    text = re.sub(r"[^0-9A-Za-z]+", "_", text).strip("_")
    return text or "unknown"


def human_readable_feature(feat: str) -> str:
    """
    Turn internal feature names like 'pair__publication__id245instance'
    into something more readable for the plots.
    Keep it short enough to fit on a bar-chart y-axis.
    """
    parts = feat.split("__")
    if len(parts) == 3:
        category, predicate, obj = parts
        predicate = predicate.replace("_", " ")
        obj = obj.replace("_", " ")
        # truncate very long object ids
        if len(obj) > 30:
            obj = obj[:27] + "..."
        return f"{category}: {predicate} → {obj}"
    elif len(parts) == 2:
        category, predicate = parts
        predicate = predicate.replace("_", " ")
        return f"{category}: {predicate}"
    return feat.replace("_", " ")


def read_tsv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", errors="ignore", newline="") as fh:
        reader = csv.DictReader(fh, delimiter="\t")
        return [dict(row) for row in reader]


def load_graph(path: Path) -> Graph:
    g = Graph()
    g.parse(path, format="n3")
    return g


# ---------------------------------------------------------------------------
# Indexing structure for fast neighbourhood lookups
# ---------------------------------------------------------------------------

@dataclass
class RDFIndex:
    outgoing: dict[object, list[tuple[object, object]]]
    incoming: dict[object, list[tuple[object, object]]]
    types: dict[object, set[object]]


def build_index(graph: Graph) -> RDFIndex:
    outgoing: dict[object, list[tuple[object, object]]] = defaultdict(list)
    incoming: dict[object, list[tuple[object, object]]] = defaultdict(list)
    types: dict[object, set[object]] = defaultdict(set)

    for s, p, o in graph:
        outgoing[s].append((p, o))
        incoming[o].append((s, p))
        if p == RDF.type:
            types[s].add(o)

    return RDFIndex(outgoing=outgoing, incoming=incoming, types=types)


# ---------------------------------------------------------------------------
# Dataset analysis
# ---------------------------------------------------------------------------

def analyze_dataset(graph: Graph, index: RDFIndex, output_dir: Path) -> dict:
    """
    Compute statistics about the RDF graph and save a summary + plots.
    This covers the "Data Analysis" section of the report.
    """
    num_triples = len(graph)

    # count unique subjects, predicates, objects
    subjects = set()
    predicates = set()
    objects = set()
    for s, p, o in graph:
        subjects.add(s)
        predicates.add(p)
        objects.add(o)

    all_nodes = subjects | {o for o in objects if isinstance(o, URIRef)}
    literal_count = sum(1 for o in objects if isinstance(o, Literal))

    # --- node type distribution ---
    type_counter: Counter[str] = Counter()
    for node, node_types in index.types.items():
        for t in node_types:
            type_counter[short_name(t)] += 1

    # --- predicate frequency ---
    pred_counter: Counter[str] = Counter()
    for s, p, o in graph:
        pred_counter[short_name(p)] += 1

    # --- degree distribution (outgoing edges per node) ---
    degrees = [len(edges) for edges in index.outgoing.values()]

    stats = {
        "num_triples": num_triples,
        "num_subjects": len(subjects),
        "num_predicates": len(predicates),
        "num_objects": len(objects),
        "num_uri_nodes": len(all_nodes),
        "num_literals": literal_count,
        "top_10_node_types": dict(type_counter.most_common(10)),
        "top_10_predicates": dict(pred_counter.most_common(10)),
        "degree_stats": {
            "mean": float(np.mean(degrees)),
            "median": float(np.median(degrees)),
            "max": int(np.max(degrees)),
            "min": int(np.min(degrees)),
        },
    }

    # ---------- plot: top node types ----------
    fig, ax = plt.subplots(figsize=(8, 4))
    top_types = type_counter.most_common(12)
    names = [t[0] for t in top_types]
    counts = [t[1] for t in top_types]
    ax.barh(names[::-1], counts[::-1], color="#4878a8")
    ax.set_xlabel("Count")
    ax.set_title("Most Frequent Node Types in AIFB")
    plt.tight_layout()
    fig.savefig(output_dir / "node_types.pdf", dpi=150)
    fig.savefig(output_dir / "node_types.png", dpi=150)
    plt.close(fig)

    # ---------- plot: top predicates ----------
    fig, ax = plt.subplots(figsize=(8, 4))
    top_preds = pred_counter.most_common(12)
    pnames = [t[0] for t in top_preds]
    pcounts = [t[1] for t in top_preds]
    ax.barh(pnames[::-1], pcounts[::-1], color="#e07b54")
    ax.set_xlabel("Count")
    ax.set_title("Most Frequent Predicates in AIFB")
    plt.tight_layout()
    fig.savefig(output_dir / "predicates.pdf", dpi=150)
    fig.savefig(output_dir / "predicates.png", dpi=150)
    plt.close(fig)

    # ---------- plot: degree distribution ----------
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(degrees, bins=50, color="#6bae6b", edgecolor="white", linewidth=0.5)
    ax.set_xlabel("Out-Degree")
    ax.set_ylabel("Number of Nodes")
    ax.set_title("Out-Degree Distribution")
    ax.set_yscale("log")
    plt.tight_layout()
    fig.savefig(output_dir / "degree_distribution.pdf", dpi=150)
    fig.savefig(output_dir / "degree_distribution.png", dpi=150)
    plt.close(fig)

    return stats


# ---------------------------------------------------------------------------
# Feature engineering: 1-hop neighbourhood of a person node
# ---------------------------------------------------------------------------

def build_features(person: str, index: RDFIndex) -> Counter[str]:
    """
    Represent a person by their 1-hop neighbourhood in the RDF graph.

    Feature types:
      type__X        : person has rdf:type X
      out__P         : person has outgoing predicate P (count)
      pair__P__O     : person --P--> O  (binary)
      objtype__P__T  : the object reached via P has type T
      literal__P     : person has a literal-valued property P
      in__P          : someone links to this person via predicate P
    """
    subj = URIRef(person)
    feats: Counter[str] = Counter()

    # rdf:type of the person
    for cls in index.types.get(subj, set()):
        feats[f"type__{short_name(cls)}"] += 1

    # outgoing edges
    for pred, obj in index.outgoing.get(subj, []):
        pname = short_name(pred)
        feats[f"out__{pname}"] += 1

        if isinstance(obj, URIRef):
            oname = short_name(obj)
            feats[f"pair__{pname}__{oname}"] += 1
            for otype in index.types.get(obj, set()):
                feats[f"objtype__{pname}__{short_name(otype)}"] += 1
        else:
            feats[f"literal__{pname}"] += 1

    # incoming edges
    for _, pred in index.incoming.get(subj, []):
        feats[f"in__{short_name(pred)}"] += 1

    return feats


# ---------------------------------------------------------------------------
# Name lookup: try to get a human-readable name from the graph for a person
# ---------------------------------------------------------------------------

def get_person_name(graph: Graph, person_uri: str) -> str:
    """Look up the :name or rdfs:label of a person; fall back to local URI."""
    uri = URIRef(person_uri)

    # try swrc:name (used in AIFB)
    name_pred = URIRef("http://swrc.ontoware.org/ontology#name")
    for _, _, name_val in graph.triples((uri, name_pred, None)):
        return str(name_val)

    # try rdfs:label
    for _, _, lbl in graph.triples((uri, RDFS.label, None)):
        return str(lbl)

    return short_name(person_uri)


# ---------------------------------------------------------------------------
# Explanation: coefficient-based contributions (white-box)
# ---------------------------------------------------------------------------

def top_feature_contributions(
    model: LogisticRegression,
    vectorizer: DictVectorizer,
    row,
    class_label: str,
    top_k: int,
) -> list[dict]:
    """
    For a single instance, return the top-k features ranked by their
    contribution to the predicted class score (weight * value).
    """
    class_idx = list(model.classes_).index(class_label)
    feat_names = vectorizer.get_feature_names_out()
    coefs = model.coef_[class_idx]

    contribs = []
    for fi, val in zip(row.indices, row.data):
        c = float(val * coefs[fi])
        contribs.append({
            "feature": feat_names[fi],
            "feature_readable": human_readable_feature(feat_names[fi]),
            "value": float(val),
            "weight": float(coefs[fi]),
            "contribution": c,
        })

    contribs.sort(key=lambda x: x["contribution"], reverse=True)
    return contribs[:top_k]


# ---------------------------------------------------------------------------
# Perturbation-based explanation (model-agnostic, similar to LIME)
# ---------------------------------------------------------------------------

def perturbation_importance(
    model, row, vectorizer: DictVectorizer, predicted_class_idx: int, n_samples: int = 200
) -> dict[str, float]:
    """
    Estimate each active feature's importance by randomly masking subsets
    of features and measuring the change in predicted probability.
    This is a simplified leave-some-out approach inspired by LIME/SHAP.
    """
    rng = np.random.RandomState(42)
    feat_names = vectorizer.get_feature_names_out()
    active_indices = row.indices.tolist()

    if len(active_indices) == 0:
        return {}

    base_prob = float(model.predict_proba(row)[0, predicted_class_idx])
    importance_accum = defaultdict(list)

    for _ in range(n_samples):
        # randomly decide which active features to keep (coin flip each)
        mask = rng.randint(0, 2, size=len(active_indices))
        perturbed = row.copy()
        for idx, keep in zip(active_indices, mask):
            if not keep:
                perturbed[0, idx] = 0.0
        perturbed.eliminate_zeros()

        new_prob = float(model.predict_proba(perturbed)[0, predicted_class_idx])
        diff = base_prob - new_prob

        # attribute the probability change to the features that were removed
        removed = [active_indices[i] for i, k in enumerate(mask) if not k]
        if removed:
            share = diff / len(removed)
            for fi in removed:
                importance_accum[feat_names[fi]].append(share)

    # average over all perturbation rounds
    avg_importance = {
        feat: float(np.mean(vals))
        for feat, vals in importance_accum.items()
    }
    return avg_importance


# ---------------------------------------------------------------------------
# Faithfulness evaluation helpers
# ---------------------------------------------------------------------------

def remove_features_from_row(row, feat_names: list[str], to_remove: Iterable[str]):
    """Zero-out specific features from a sparse row."""
    updated = row.copy()
    name2idx = {n: i for i, n in enumerate(feat_names)}
    for f in to_remove:
        fi = name2idx.get(f)
        if fi is not None:
            updated[0, fi] = 0
    updated.eliminate_zeros()
    return updated


# ---------------------------------------------------------------------------
# Plotting helpers
# ---------------------------------------------------------------------------

def plot_confusion_matrix(y_true, y_pred, classes, output_path: Path, title="Confusion Matrix"):
    """Standard confusion matrix heatmap."""
    labels = [readable_label(c) for c in classes]
    cm = confusion_matrix(y_true, y_pred, labels=classes)

    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=labels, yticklabels=labels, ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(title)
    plt.tight_layout()
    fig.savefig(output_path.with_suffix(".pdf"), dpi=150)
    fig.savefig(output_path.with_suffix(".png"), dpi=150)
    plt.close(fig)


def plot_explanation_bar(
    contributions: list[dict],
    person_name: str,
    true_label: str,
    pred_label: str,
    output_path: Path,
):
    """Horizontal bar chart showing feature contributions for one instance."""
    feats = [c["feature_readable"] for c in contributions][::-1]
    vals = [c["contribution"] for c in contributions][::-1]
    colors = ["#4878a8" if v >= 0 else "#d9534f" for v in vals]

    fig, ax = plt.subplots(figsize=(8, max(3, 0.45 * len(feats))))
    ax.barh(feats, vals, color=colors)
    ax.axvline(0, color="black", linewidth=0.5)
    ax.set_xlabel("Contribution to predicted class score")
    ax.set_title(f"{person_name}\nTrue: {true_label} | Predicted: {pred_label}")
    plt.tight_layout()
    fig.savefig(output_path.with_suffix(".pdf"), dpi=150)
    fig.savefig(output_path.with_suffix(".png"), dpi=150)
    plt.close(fig)


def plot_faithfulness_histogram(changes: list[float], output_path: Path):
    """Distribution of confidence drops when top features are removed."""
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(changes, bins=25, color="#8b5fbf", edgecolor="white", linewidth=0.5)
    ax.axvline(np.mean(changes), color="red", linestyle="--", label=f"Mean = {np.mean(changes):.3f}")
    ax.axvline(np.median(changes), color="orange", linestyle="--", label=f"Median = {np.median(changes):.3f}")
    ax.set_xlabel("Confidence Drop After Feature Deletion")
    ax.set_ylabel("Count")
    ax.set_title("Faithfulness: Confidence Drop Distribution")
    ax.legend()
    plt.tight_layout()
    fig.savefig(output_path.with_suffix(".pdf"), dpi=150)
    fig.savefig(output_path.with_suffix(".png"), dpi=150)
    plt.close(fig)


def plot_model_comparison(results: dict, output_path: Path):
    """Grouped bar chart comparing models on test metrics."""
    model_names = list(results.keys())
    metric_names = ["Accuracy", "Macro F1", "Weighted F1"]

    x = np.arange(len(metric_names))
    width = 0.25
    colors = ["#4878a8", "#e07b54", "#6bae6b"]

    fig, ax = plt.subplots(figsize=(7, 4))
    for i, model_name in enumerate(model_names):
        vals = [
            results[model_name]["accuracy"],
            results[model_name]["macro_f1"],
            results[model_name]["weighted_f1"],
        ]
        ax.bar(x + i * width, vals, width, label=model_name, color=colors[i % len(colors)])

    ax.set_ylabel("Score")
    ax.set_title("Model Comparison on Test Set")
    ax.set_xticks(x + width)
    ax.set_xticklabels(metric_names)
    ax.set_ylim(0, 1.05)
    ax.legend()
    plt.tight_layout()
    fig.savefig(output_path.with_suffix(".pdf"), dpi=150)
    fig.savefig(output_path.with_suffix(".png"), dpi=150)
    plt.close(fig)


def plot_label_distribution(train_labels, test_labels, classes, output_path: Path):
    """Show class distribution in train vs test."""
    train_counts = Counter(train_labels)
    test_counts = Counter(test_labels)
    labels = [readable_label(c) for c in classes]

    x = np.arange(len(classes))
    width = 0.35

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(x - width / 2, [train_counts.get(c, 0) for c in classes], width,
           label="Train", color="#4878a8")
    ax.bar(x + width / 2, [test_counts.get(c, 0) for c in classes], width,
           label="Test", color="#e07b54")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=20, ha="right", fontsize=9)
    ax.set_ylabel("Count")
    ax.set_title("Label Distribution: Train vs. Test")
    ax.legend()
    plt.tight_layout()
    fig.savefig(output_path.with_suffix(".pdf"), dpi=150)
    fig.savefig(output_path.with_suffix(".png"), dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Sparse matrix compatibility fix (Python 3.12+ / newer scipy)
# ---------------------------------------------------------------------------

def to_32bit_sparse(matrix):
    matrix = matrix.tocsr()
    matrix.indices = matrix.indices.astype(np.int32, copy=False)
    matrix.indptr = matrix.indptr.astype(np.int32, copy=False)
    return matrix


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="AIFB RDF classification + explanation pipeline (Strategy 2a)"
    )
    parser.add_argument(
        "--data-dir", type=Path,
        default=Path(__file__).resolve().parent,
        help="Directory with aifbfixed_complete.n3, trainingSet.tsv, testSet.tsv",
    )
    parser.add_argument(
        "--output-dir", type=Path,
        default=Path(__file__).resolve().parent / "results",
        help="Where to write outputs (JSON + plots)",
    )
    parser.add_argument("--top-k", type=int, default=8,
                        help="Number of top features in explanations")
    parser.add_argument("--deletion-k", type=int, default=5,
                        help="Number of features to remove for faithfulness test")
    parser.add_argument("--perturbation-samples", type=int, default=300,
                        help="Number of perturbation rounds for importance estimation")
    args = parser.parse_args()

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    # ======================================================================
    # 1. Load RDF graph
    # ======================================================================
    print("Loading RDF graph...")
    graph_path = args.data_dir / "aifbfixed_complete.n3"
    train_path = args.data_dir / "trainingSet.tsv"
    test_path = args.data_dir / "testSet.tsv"

    graph = load_graph(graph_path)
    index = build_index(graph)
    print(f"  Graph loaded: {len(graph)} triples")

    # ======================================================================
    # 2. Dataset analysis
    # ======================================================================
    print("Analyzing dataset...")
    dataset_stats = analyze_dataset(graph, index, output_dir)
    (output_dir / "dataset_stats.json").write_text(
        json.dumps(dataset_stats, indent=2), encoding="utf-8"
    )
    print(f"  Unique URI nodes: {dataset_stats['num_uri_nodes']}")
    print(f"  Distinct predicates: {dataset_stats['num_predicates']}")
    print(f"  Literals: {dataset_stats['num_literals']}")

    # ======================================================================
    # 3. Prepare train / test splits
    # ======================================================================
    train_rows = read_tsv(train_path)
    test_rows = read_tsv(test_path)
    if not train_rows or not test_rows:
        raise ValueError("Empty split file — check data-dir")

    train_persons, train_features = zip(
        *[(r["person"], build_features(r["person"], index)) for r in train_rows]
    )
    test_persons, test_features = zip(
        *[(r["person"], build_features(r["person"], index)) for r in test_rows]
    )

    train_labels = [r["label_affiliation"] for r in train_rows]
    test_labels = [r["label_affiliation"] for r in test_rows]

    print(f"  Train: {len(train_rows)} instances, Test: {len(test_rows)} instances")

    # vectorize
    vectorizer = DictVectorizer(sparse=True)
    x_train = to_32bit_sparse(vectorizer.fit_transform(train_features))
    x_test = to_32bit_sparse(vectorizer.transform(test_features))
    feat_names = vectorizer.get_feature_names_out().tolist()
    print(f"  Feature dimensionality: {len(feat_names)}")

    # figure out which classes actually appear in the data
    all_classes = sorted(set(train_labels) | set(test_labels))

    # label distribution plot
    plot_label_distribution(train_labels, test_labels, all_classes, output_dir / "label_distribution")

    # ======================================================================
    # 4. Model training
    # ======================================================================
    print("Training models...")

    # --- Logistic Regression (our main model) ---
    lr = LogisticRegression(max_iter=2000, solver="lbfgs", random_state=42)
    lr.fit(x_train, train_labels)

    # --- Decision Tree (interpretable comparison) ---
    dt = DecisionTreeClassifier(max_depth=8, random_state=42, min_samples_leaf=3)
    dt.fit(x_train, train_labels)

    # --- Majority baseline ---
    dummy = DummyClassifier(strategy="most_frequent", random_state=42)
    dummy.fit(x_train, train_labels)

    # ======================================================================
    # 5. Evaluation
    # ======================================================================
    print("Evaluating models...")

    def eval_model(model, name):
        tr_pred = model.predict(x_train)
        te_pred = model.predict(x_test)
        return {
            "train": {
                "accuracy": accuracy_score(train_labels, tr_pred),
                "macro_f1": f1_score(train_labels, tr_pred, average="macro", zero_division=0),
                "weighted_f1": f1_score(train_labels, tr_pred, average="weighted", zero_division=0),
            },
            "test": {
                "accuracy": accuracy_score(test_labels, te_pred),
                "macro_f1": f1_score(test_labels, te_pred, average="macro", zero_division=0),
                "weighted_f1": f1_score(test_labels, te_pred, average="weighted", zero_division=0),
            },
        }

    lr_metrics = eval_model(lr, "LogReg")
    dt_metrics = eval_model(dt, "DecisionTree")
    dummy_metrics = eval_model(dummy, "Majority")

    metrics = {
        "dataset": dataset_stats,
        "logistic_regression": lr_metrics,
        "decision_tree": dt_metrics,
        "majority_baseline": dummy_metrics,
    }

    for name, m in [("Logistic Regression", lr_metrics),
                    ("Decision Tree", dt_metrics),
                    ("Majority Baseline", dummy_metrics)]:
        print(f"  {name:25s}  test acc={m['test']['accuracy']:.3f}  "
              f"macro-F1={m['test']['macro_f1']:.3f}")

    # per-class report (logistic regression)
    lr_test_pred = lr.predict(x_test)
    report_dict = classification_report(
        test_labels, lr_test_pred, output_dict=True, zero_division=0
    )
    # rewrite keys to human-readable names
    class_report = {}
    for cls in lr.classes_:
        if cls in report_dict:
            class_report[readable_label(cls)] = {
                k: float(v) for k, v in report_dict[cls].items()
                if k in ("precision", "recall", "f1-score", "support")
            }
    class_report["accuracy"] = float(report_dict["accuracy"])

    # confusion matrix
    plot_confusion_matrix(
        test_labels, lr_test_pred, list(lr.classes_),
        output_dir / "confusion_matrix",
        title="Logistic Regression — Test Confusion Matrix",
    )

    # model comparison plot
    plot_model_comparison(
        {
            "Logistic Regression": lr_metrics["test"],
            "Decision Tree": dt_metrics["test"],
            "Majority Baseline": dummy_metrics["test"],
        },
        output_dir / "model_comparison",
    )

    # ======================================================================
    # 6. Decision tree textual rules (global explanation)
    # ======================================================================
    print("Extracting decision tree rules...")
    dt_rules = export_text(
        dt, feature_names=[human_readable_feature(f) for f in feat_names], max_depth=5
    )
    (output_dir / "decision_tree_rules.txt").write_text(dt_rules, encoding="utf-8")

    # ======================================================================
    # 7. Local explanations + faithfulness evaluation
    # ======================================================================
    print("Generating explanations and evaluating faithfulness...")

    test_proba = lr.predict_proba(x_test)
    explanations = []
    faithfulness_drops = []
    prediction_flips = 0
    perturbation_explanations = []

    for i, (person, label) in enumerate(zip(test_persons, test_labels)):
        row = x_test[i]
        pred_idx = int(test_proba[i].argmax())
        pred_label = lr.classes_[pred_idx]
        pred_prob = float(test_proba[i][pred_idx])

        # --- coefficient-based explanation ---
        coeff_expl = top_feature_contributions(lr, vectorizer, row, pred_label, args.top_k)

        # --- faithfulness: delete top-k features and re-predict ---
        top_feats = [c["feature"] for c in coeff_expl[:args.deletion_k]]
        modified = remove_features_from_row(row, feat_names, top_feats)
        mod_prob = float(lr.predict_proba(modified)[0, pred_idx])
        drop = pred_prob - mod_prob
        faithfulness_drops.append(drop)
        if lr.predict(modified)[0] != pred_label:
            prediction_flips += 1

        # --- perturbation-based importance (for a subset to save time) ---
        if i < 10:
            perturb_imp = perturbation_importance(
                lr, row, vectorizer, pred_idx, n_samples=args.perturbation_samples
            )
            # pick top-k
            sorted_pi = sorted(perturb_imp.items(), key=lambda x: x[1], reverse=True)[:args.top_k]
            perturbation_explanations.append({
                "person": person,
                "person_name": get_person_name(graph, person),
                "top_features": [
                    {"feature": f, "feature_readable": human_readable_feature(f), "importance": imp}
                    for f, imp in sorted_pi
                ],
            })

        # store detailed explanation for the first few instances
        if i < 10:
            person_name = get_person_name(graph, person)
            expl_entry = {
                "person": person,
                "person_name": person_name,
                "true_label": readable_label(label),
                "predicted_label": readable_label(pred_label),
                "predicted_probability": round(pred_prob, 4),
                "correct": label == pred_label,
                "confidence_drop_after_deletion": round(drop, 4),
                "top_features": coeff_expl,
            }
            explanations.append(expl_entry)

            # plot individual explanation
            plot_explanation_bar(
                coeff_expl, person_name,
                readable_label(label), readable_label(pred_label),
                output_dir / f"explanation_{i}",
            )

    faithfulness_eval = {
        "method": "feature_deletion (top-k coefficient contributions)",
        "deletion_k": args.deletion_k,
        "num_test_instances": len(faithfulness_drops),
        "avg_confidence_drop": round(float(np.mean(faithfulness_drops)), 4),
        "median_confidence_drop": round(float(np.median(faithfulness_drops)), 4),
        "std_confidence_drop": round(float(np.std(faithfulness_drops)), 4),
        "prediction_flip_rate": round(prediction_flips / len(faithfulness_drops), 4),
        "max_confidence_drop": round(float(np.max(faithfulness_drops)), 4),
        "min_confidence_drop": round(float(np.min(faithfulness_drops)), 4),
    }

    plot_faithfulness_histogram(faithfulness_drops, output_dir / "faithfulness_histogram")

    # ======================================================================
    # 8. Comparison: coefficient vs perturbation explanations
    # ======================================================================
    # For the first few test instances, see how well the two methods agree
    agreement_scores = []
    for coeff_expl, pert_expl in zip(explanations, perturbation_explanations):
        coeff_top = set(c["feature"] for c in coeff_expl["top_features"][:5])
        pert_top = set(p["feature"] for p in pert_expl["top_features"][:5])
        if coeff_top or pert_top:
            jaccard = len(coeff_top & pert_top) / len(coeff_top | pert_top)
            agreement_scores.append(jaccard)

    explanation_agreement = {
        "method": "Jaccard similarity between top-5 coefficient and perturbation features",
        "scores": [round(s, 3) for s in agreement_scores],
        "mean_jaccard": round(float(np.mean(agreement_scores)), 3) if agreement_scores else 0,
    }

    # ======================================================================
    # 9. Save all outputs
    # ======================================================================
    print("Saving results...")

    (output_dir / "metrics.json").write_text(
        json.dumps(metrics, indent=2), encoding="utf-8"
    )
    (output_dir / "classification_report.json").write_text(
        json.dumps(class_report, indent=2), encoding="utf-8"
    )
    (output_dir / "explanations.json").write_text(
        json.dumps(explanations, indent=2, default=str), encoding="utf-8"
    )
    (output_dir / "perturbation_explanations.json").write_text(
        json.dumps(perturbation_explanations, indent=2, default=str), encoding="utf-8"
    )
    (output_dir / "faithfulness_evaluation.json").write_text(
        json.dumps(faithfulness_eval, indent=2), encoding="utf-8"
    )
    (output_dir / "explanation_agreement.json").write_text(
        json.dumps(explanation_agreement, indent=2), encoding="utf-8"
    )

    # ======================================================================
    # 10. Print summary to console
    # ======================================================================
    print("\n" + "=" * 60)
    print("RESULTS SUMMARY")
    print("=" * 60)
    print(f"\nDataset: AIFB ({len(graph)} triples, {len(feat_names)} features)")
    print(f"Train: {len(train_rows)} | Test: {len(test_rows)}\n")

    print("Test Performance:")
    print(f"  {'Model':<25s} {'Accuracy':>10s} {'Macro-F1':>10s} {'W-F1':>10s}")
    print(f"  {'-'*55}")
    for name, m in [("Logistic Regression", lr_metrics),
                    ("Decision Tree", dt_metrics),
                    ("Majority Baseline", dummy_metrics)]:
        print(f"  {name:<25s} {m['test']['accuracy']:>10.3f} "
              f"{m['test']['macro_f1']:>10.3f} {m['test']['weighted_f1']:>10.3f}")

    print(f"\nFaithfulness (feature deletion, k={args.deletion_k}):")
    print(f"  Avg confidence drop:  {faithfulness_eval['avg_confidence_drop']:.4f}")
    print(f"  Prediction flip rate: {faithfulness_eval['prediction_flip_rate']:.4f}")

    print(f"\nExplanation agreement (coeff. vs perturbation):")
    print(f"  Mean Jaccard@5: {explanation_agreement['mean_jaccard']:.3f}")

    print(f"\nOutputs saved to: {output_dir.resolve()}")
    print("=" * 60)


if __name__ == "__main__":
    main()
