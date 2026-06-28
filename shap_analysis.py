"""
shap_analysis.py
================
Tejas Khopade's contribution to the XAI Mini Project.
Adds SHAP analysis on top of the existing Logistic Regression pipeline.

Usage:
    python shap_analysis.py
"""

from __future__ import annotations
import json
import warnings
from collections import Counter, defaultdict
from pathlib import Path
import re
import csv

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np
import shap
from scipy.stats import spearmanr

from rdflib import Graph, Literal, URIRef
from rdflib.namespace import RDF, RDFS
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression

warnings.filterwarnings("ignore")

RESEARCH_GROUP_LABELS = {
    "http://www.aifb.uni-karlsruhe.de/Forschungsgruppen/viewForschungsgruppeOWL/id1instance": "Business Information Systems",
    "http://www.aifb.uni-karlsruhe.de/Forschungsgruppen/viewForschungsgruppeOWL/id2instance": "Efficient Algorithms",
    "http://www.aifb.uni-karlsruhe.de/Forschungsgruppen/viewForschungsgruppeOWL/id3instance": "Knowledge Management",
    "http://www.aifb.uni-karlsruhe.de/Forschungsgruppen/viewForschungsgruppeOWL/id4instance": "Complexity Management",
    "http://www.aifb.uni-karlsruhe.de/Forschungsgruppen/viewForschungsgruppeOWL/id5instance": "Usability Engineering",
}

def readable_label(uri):
    return RESEARCH_GROUP_LABELS.get(uri, short_name(uri))

def short_name(value):
    text = str(value)
    if "#" in text:
        text = text.rsplit("#", 1)[-1]
    if "/" in text:
        text = text.rsplit("/", 1)[-1]
    text = re.sub(r"[^0-9A-Za-z]+", "_", text).strip("_")
    return text or "unknown"

def human_readable_feature(feat):
    parts = feat.split("__")
    if len(parts) == 3:
        _, predicate, obj = parts
        predicate = predicate.replace("_", " ")
        obj = obj.replace("_", " ")
        if len(obj) > 25:
            obj = obj[:22] + "..."
        return f"{predicate} -> {obj}"
    elif len(parts) == 2:
        _, predicate = parts
        return predicate.replace("_", " ")
    return feat.replace("_", " ")

def load_graph(path):
    g = Graph()
    g.parse(path, format="n3")
    return g

def build_index(graph):
    outgoing = defaultdict(list)
    incoming = defaultdict(list)
    types = defaultdict(set)
    for s, p, o in graph:
        outgoing[s].append((p, o))
        incoming[o].append((s, p))
        if p == RDF.type:
            types[s].add(o)
    return outgoing, incoming, types

def build_features(person, outgoing, incoming, types):
    subj = URIRef(person)
    feats = Counter()
    for cls in types.get(subj, set()):
        feats[f"type__{short_name(cls)}"] += 1
    for pred, obj in outgoing.get(subj, []):
        pname = short_name(pred)
        feats[f"out__{pname}"] += 1
        if isinstance(obj, URIRef):
            feats[f"pair__{pname}__{short_name(obj)}"] += 1
            for otype in types.get(obj, set()):
                feats[f"objtype__{pname}__{short_name(otype)}"] += 1
        else:
            feats[f"literal__{pname}"] += 1
    for _, pred in incoming.get(subj, []):
        feats[f"in__{short_name(pred)}"] += 1
    return feats

def get_person_name(graph, person_uri):
    uri = URIRef(person_uri)
    name_pred = URIRef("http://swrc.ontoware.org/ontology#name")
    for _, _, name_val in graph.triples((uri, name_pred, None)):
        return str(name_val)
    for _, _, lbl in graph.triples((uri, RDFS.label, None)):
        return str(lbl)
    return short_name(person_uri)

def read_tsv(path):
    with open(path, "r", encoding="utf-8", errors="ignore", newline="") as fh:
        reader = csv.DictReader(fh, delimiter="\t")
        return [dict(row) for row in reader]

def get_shap_for_class(shap_values, class_idx, sample_idx=None):
    """Safely extract SHAP values for a given class index."""
    if isinstance(shap_values, list):
        sv = shap_values[class_idx]  # shape: (n_samples, n_features)
    elif isinstance(shap_values, np.ndarray) and shap_values.ndim == 3:
        sv = shap_values[:, :, class_idx]  # shape: (n_samples, n_features)
    else:
        sv = shap_values  # shape: (n_samples, n_features)
    if sample_idx is not None:
        return sv[sample_idx]  # shape: (n_features,)
    return sv  # shape: (n_samples, n_features)

def run_shap_analysis(data_dir=Path("."), output_dir=Path("results/shap")):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    data_dir = Path(data_dir)

    # --- Load data ---
    print("Loading AIFB graph...")
    graph = load_graph(data_dir / "aifbfixed_complete.n3")
    outgoing, incoming, types = build_index(graph)

    train_rows = read_tsv(data_dir / "trainingSet.tsv")
    test_rows  = read_tsv(data_dir / "testSet.tsv")

    person_col = next(k for k in train_rows[0] if "person" in k.lower() or "entity" in k.lower())
    label_col  = next(k for k in train_rows[0] if k != person_col)

    train_persons = [r[person_col] for r in train_rows]
    train_labels  = [r[label_col]  for r in train_rows]
    test_persons  = [r[person_col] for r in test_rows]
    test_labels   = [r[label_col]  for r in test_rows]

    # --- Features ---
    print("Building features...")
    train_feats = [build_features(p, outgoing, incoming, types) for p in train_persons]
    test_feats  = [build_features(p, outgoing, incoming, types) for p in test_persons]

    vectorizer = DictVectorizer(sparse=True)
    x_train = vectorizer.fit_transform(train_feats)
    x_test  = vectorizer.transform(test_feats)
    feat_names    = list(vectorizer.get_feature_names_out())
    readable_names = [human_readable_feature(f) for f in feat_names]

    # --- Train model ---
    print("Training Logistic Regression...")
    lr = LogisticRegression(max_iter=1000, C=1.0, solver="lbfgs",
                             multi_class="multinomial", random_state=42)
    lr.fit(x_train, train_labels)
    classes = list(lr.classes_)
    n_classes = len(classes)

    # --- SHAP ---
    print("Computing SHAP values (this may take ~30 seconds)...")
    x_train_dense = x_train.toarray()
    x_test_dense  = x_test.toarray()

    explainer   = shap.LinearExplainer(lr, x_train_dense, feature_perturbation="interventional")
    shap_values = explainer.shap_values(x_test_dense)

    # --- Plot 1: Global summary ---
    print("Generating global SHAP summary plot...")
    global_importance = np.zeros(len(feat_names))
    for ci in range(n_classes):
        sv = get_shap_for_class(shap_values, ci)
        global_importance += np.abs(sv).mean(axis=0)
    global_importance /= n_classes

    top_idx  = np.argsort(global_importance)[-15:]
    top_vals = [float(global_importance[i]) for i in top_idx]
    top_names = [readable_names[i] for i in top_idx]

    fig, ax = plt.subplots(figsize=(9, 6))
    ax.barh(range(len(top_idx)), top_vals, color="#4878a8")
    ax.set_yticks(range(len(top_idx)))
    ax.set_yticklabels(top_names, fontsize=9)
    ax.set_xlabel("Mean |SHAP value| across all classes")
    ax.set_title("Global Feature Importance (SHAP)\nTop 15 features — AIFB Research Group Classification")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()
    fig.savefig(output_dir / "shap_global_summary.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  Saved: shap_global_summary.png")

    # --- Plot 2: Per-class ---
    print("Generating per-class SHAP plots...")
    fig, axes = plt.subplots(1, n_classes, figsize=(4 * n_classes, 6), sharey=False)

    for ci in range(n_classes):
        sv = get_shap_for_class(shap_values, ci)
        class_importance = np.abs(sv).mean(axis=0)
        top_idx_c  = list(np.argsort(class_importance)[-10:])
        top_vals_c = [float(class_importance[i]) for i in top_idx_c]
        top_names_c = [readable_names[i] for i in top_idx_c]

        ax = axes[ci]
        ax.barh(range(len(top_idx_c)), top_vals_c, color="#e07b54")
        ax.set_yticks(range(len(top_idx_c)))
        ax.set_yticklabels(top_names_c, fontsize=7)
        ax.set_title(readable_label(classes[ci]), fontsize=9, fontweight="bold")
        ax.set_xlabel("Mean |SHAP|", fontsize=8)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    fig.suptitle("Per-Class SHAP Feature Importance — AIFB Dataset", fontsize=11)
    plt.tight_layout()
    fig.savefig(output_dir / "shap_per_class.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  Saved: shap_per_class.png")

    # --- Plot 3: Waterfall ---
    print("Generating single-instance waterfall plot...")
    test_preds  = lr.predict(x_test)
    correct_idx = next((i for i, (p, l) in enumerate(zip(test_preds, test_labels)) if p == l), 0)

    person_name    = get_person_name(graph, test_persons[correct_idx])
    true_label     = readable_label(test_labels[correct_idx])
    pred_label     = readable_label(test_preds[correct_idx])
    pred_class_idx = classes.index(test_preds[correct_idx])  # 0,1,2,3,4

    instance_shap = get_shap_for_class(shap_values, pred_class_idx, correct_idx)
    base_val = (explainer.expected_value[pred_class_idx]
                if hasattr(explainer.expected_value, "__len__")
                else float(explainer.expected_value))

    top_n     = 12
    top_idx_w = list(np.argsort(np.abs(instance_shap))[-top_n:])
    w_vals    = np.array([float(instance_shap[i]) for i in top_idx_w])
    w_names   = [readable_names[i] for i in top_idx_w]

    sort_order = list(np.argsort(w_vals))
    w_vals  = np.array([w_vals[i] for i in sort_order])
    w_names = [w_names[i] for i in sort_order]
    colors  = ["#d62728" if v < 0 else "#2ca02c" for v in w_vals]

    fig, ax = plt.subplots(figsize=(9, 6))
    ax.barh(range(len(w_vals)), w_vals, color=colors)
    ax.set_yticks(range(len(w_vals)))
    ax.set_yticklabels(w_names, fontsize=9)
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_xlabel("SHAP value (contribution to predicted class score)")
    ax.set_title(f"SHAP Explanation for: {person_name}\nTrue: {true_label}  |  Predicted: {pred_label}", fontsize=10)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    legend_elements = [
        Patch(facecolor="#2ca02c", label="Increases prediction"),
        Patch(facecolor="#d62728", label="Decreases prediction"),
    ]
    ax.legend(handles=legend_elements, loc="lower right", fontsize=8)
    plt.tight_layout()
    fig.savefig(output_dir / "shap_waterfall_example.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: shap_waterfall_example.png  (person: {person_name})")

    # --- SHAP vs Coefficient comparison ---
    print("Comparing SHAP vs coefficient explanations...")
    correlations = []
    for ci in range(n_classes):
        sv = get_shap_for_class(shap_values, ci)
        class_importance = np.abs(sv).mean(axis=0)
        coef = lr.coef_[ci]
        corr, pval = spearmanr(np.abs(coef), class_importance)
        correlations.append({
            "class": readable_label(classes[ci]),
            "spearman_r": round(float(corr), 4),
            "p_value": round(float(pval), 6),
        })
        print(f"  {readable_label(classes[ci]):35s}  r={corr:.3f}  p={pval:.4f}")

    (output_dir / "shap_vs_coeff_correlation.json").write_text(
        json.dumps(correlations, indent=2), encoding="utf-8")

    # --- Top features per class JSON ---
    top_features_per_class = {}
    for ci in range(n_classes):
        sv = get_shap_for_class(shap_values, ci)
        class_importance = np.abs(sv).mean(axis=0)
        top_idx_c = list(np.argsort(class_importance)[-10:][::-1])
        top_features_per_class[readable_label(classes[ci])] = [
            {"feature": feat_names[i],
             "feature_readable": readable_names[i],
             "mean_abs_shap": round(float(class_importance[i]), 5)}
            for i in top_idx_c
        ]

    (output_dir / "shap_top_features_per_class.json").write_text(
        json.dumps(top_features_per_class, indent=2), encoding="utf-8")

    # --- Summary ---
    print("\n" + "=" * 55)
    print("SHAP ANALYSIS COMPLETE")
    print("=" * 55)
    print(f"Outputs saved to: {output_dir.resolve()}")
    for f in sorted(output_dir.iterdir()):
        print(f"  {f.name}")
    print("=" * 55)


if __name__ == "__main__":
    run_shap_analysis(data_dir=Path("."), output_dir=Path("results/shap"))
