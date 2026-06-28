from __future__ import annotations

import argparse
import csv
import json
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np

from rdflib import Graph, URIRef
from rdflib.namespace import RDF
from sklearn.dummy import DummyClassifier
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, f1_score


def short_name(value: object) -> str:
    """Convert an RDF term or URI into a compact feature-friendly token."""

    text = str(value)
    if "#" in text:
        text = text.rsplit("#", 1)[-1]
    if "/" in text:
        text = text.rsplit("/", 1)[-1]
    text = re.sub(r"[^0-9A-Za-z]+", "_", text).strip("_")
    return text or "unknown"


def read_tsv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", errors="ignore", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        return [dict(row) for row in reader]


def load_graph(path: Path) -> Graph:
    graph = Graph()
    graph.parse(path, format="n3")
    return graph


@dataclass
class RDFIndex:
    outgoing: dict[object, list[tuple[object, object]]]
    incoming: dict[object, list[tuple[object, object]]]
    types: dict[object, set[object]]


def build_index(graph: Graph) -> RDFIndex:
    outgoing: dict[object, list[tuple[object, object]]] = defaultdict(list)
    incoming: dict[object, list[tuple[object, object]]] = defaultdict(list)
    types: dict[object, set[object]] = defaultdict(set)

    for subject, predicate, obj in graph:
        outgoing[subject].append((predicate, obj))
        incoming[obj].append((subject, predicate))
        if predicate == RDF.type:
            types[subject].add(obj)

    return RDFIndex(outgoing=outgoing, incoming=incoming, types=types)


def build_features(person: str, index: RDFIndex) -> Counter[str]:
    """Build a simple 1-hop neighborhood representation for one person."""

    subject = URIRef(person)
    features: Counter[str] = Counter()

    for class_uri in index.types.get(subject, set()):
        features[f"type__{short_name(class_uri)}"] += 1

    for predicate, obj in index.outgoing.get(subject, []):
        predicate_name = short_name(predicate)
        features[f"out__{predicate_name}"] += 1

        if isinstance(obj, URIRef):
            object_name = short_name(obj)
            features[f"pair__{predicate_name}__{object_name}"] += 1

            for object_type in index.types.get(obj, set()):
                features[f"objtype__{predicate_name}__{short_name(object_type)}"] += 1
        else:
            features[f"literal__{predicate_name}"] += 1

    for _, predicate in index.incoming.get(subject, []):
        features[f"in__{short_name(predicate)}"] += 1

    return features


def load_split(path: Path) -> list[dict[str, str]]:
    rows = read_tsv(path)
    if not rows:
        raise ValueError(f"Empty split file: {path}")
    return rows


def vectorize_samples(samples: list[dict[str, str]], index: RDFIndex) -> tuple[list[str], list[Counter[str]]]:
    persons = [row["person"] for row in samples]
    feature_dicts = [build_features(person, index) for person in persons]
    return persons, feature_dicts


def top_feature_contributions(model: LogisticRegression, vectorizer: DictVectorizer, row, class_label: str, top_k: int) -> list[dict[str, object]]:
    class_index = list(model.classes_).index(class_label)
    feature_names = vectorizer.get_feature_names_out()
    coefficients = model.coef_[class_index]
    indices = row.indices
    values = row.data

    contributions = []
    for feature_index, value in zip(indices, values):
        weight = float(value * coefficients[feature_index])
        contributions.append(
            {
                "feature": feature_names[feature_index],
                "value": float(value),
                "weight": float(coefficients[feature_index]),
                "contribution": weight,
            }
        )

    contributions.sort(key=lambda item: item["contribution"], reverse=True)
    return contributions[:top_k]


def remove_features_from_row(row, feature_names: list[str], features_to_remove: Iterable[str]):
    updated = row.copy()
    index_by_name = {name: idx for idx, name in enumerate(feature_names)}
    for feature in features_to_remove:
        feature_index = index_by_name.get(feature)
        if feature_index is not None:
            updated[0, feature_index] = 0
    updated.eliminate_zeros()
    return updated


def ensure_directory(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def to_32bit_sparse(matrix):
    """Make sparse matrices compatible with scikit-learn on Python 3.14."""

    matrix = matrix.tocsr()
    matrix.indices = matrix.indices.astype(np.int32, copy=False)
    matrix.indptr = matrix.indptr.astype(np.int32, copy=False)
    return matrix


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a small RDF classification and explanation pipeline on the AIFB dataset.")
    parser.add_argument("--data-dir", type=Path, default=Path(__file__).resolve().parent, help="Folder containing the RDF graph and TSV splits.")
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).resolve().parent / "results", help="Folder for generated outputs.")
    parser.add_argument("--top-k", type=int, default=5, help="Number of explanation features to keep per instance.")
    parser.add_argument("--deletion-k", type=int, default=5, help="How many top features to remove for the explanation faithfulness test.")
    args = parser.parse_args()

    graph_path = args.data_dir / "aifbfixed_complete.n3"
    train_path = args.data_dir / "trainingSet.tsv"
    test_path = args.data_dir / "testSet.tsv"

    graph = load_graph(graph_path)
    index = build_index(graph)

    train_rows = load_split(train_path)
    test_rows = load_split(test_path)

    train_persons, train_features = vectorize_samples(train_rows, index)
    test_persons, test_features = vectorize_samples(test_rows, index)

    train_labels = [row["label_affiliation"] for row in train_rows]
    test_labels = [row["label_affiliation"] for row in test_rows]

    vectorizer = DictVectorizer(sparse=True)
    x_train = to_32bit_sparse(vectorizer.fit_transform(train_features))
    x_test = to_32bit_sparse(vectorizer.transform(test_features))

    model = LogisticRegression(max_iter=2000, solver="lbfgs", random_state=42)
    model.fit(x_train, train_labels)

    dummy = DummyClassifier(strategy="most_frequent", random_state=42)
    dummy.fit(x_train, train_labels)

    train_predictions = model.predict(x_train)
    test_predictions = model.predict(x_test)
    test_probabilities = model.predict_proba(x_test)
    dummy_predictions = dummy.predict(x_test)
    dummy_train_predictions = dummy.predict(x_train)

    metrics = {
        "dataset": {
            "triples": len(graph),
            "training_instances": len(train_rows),
            "test_instances": len(test_rows),
            "classes": len(model.classes_),
        },
        "model": {
            "train": {
                "accuracy": accuracy_score(train_labels, train_predictions),
                "macro_f1": f1_score(train_labels, train_predictions, average="macro"),
                "weighted_f1": f1_score(train_labels, train_predictions, average="weighted"),
            },
            "test": {
                "accuracy": accuracy_score(test_labels, test_predictions),
                "macro_f1": f1_score(test_labels, test_predictions, average="macro"),
                "weighted_f1": f1_score(test_labels, test_predictions, average="weighted"),
            },
        },
        "baseline": {
            "train_accuracy": accuracy_score(train_labels, dummy_train_predictions),
            "test_accuracy": accuracy_score(test_labels, dummy_predictions),
        },
    }

    report = classification_report(test_labels, test_predictions, output_dict=True, zero_division=0)

    feature_names = vectorizer.get_feature_names_out().tolist()
    explanations: list[dict[str, object]] = []
    faithfulness_changes: list[float] = []
    prediction_flips = 0

    for index_in_test, (person, row, label, probability_row) in enumerate(zip(test_persons, x_test, test_labels, test_probabilities)):
        predicted_index = int(probability_row.argmax())
        predicted_label = model.classes_[predicted_index]
        predicted_probability = float(probability_row[predicted_index])
        instance_explanation = top_feature_contributions(model, vectorizer, row, predicted_label, args.top_k)

        explanation_features = [item["feature"] for item in instance_explanation[: args.deletion_k]]
        modified_row = remove_features_from_row(row, feature_names, explanation_features)
        modified_probability = float(model.predict_proba(modified_row)[0, predicted_index])
        faithfulness_changes.append(predicted_probability - modified_probability)
        if model.predict(modified_row)[0] != predicted_label:
            prediction_flips += 1

        if index_in_test < 5:
            explanations.append(
                {
                    "person": person,
                    "true_label": label,
                    "predicted_label": predicted_label,
                    "predicted_probability": predicted_probability,
                    "top_features": instance_explanation,
                }
            )

    explanation_evaluation = {
        "average_confidence_drop": sum(faithfulness_changes) / len(faithfulness_changes),
        "median_confidence_drop": sorted(faithfulness_changes)[len(faithfulness_changes) // 2],
        "prediction_flip_rate": prediction_flips / len(faithfulness_changes),
    }

    class_tokens = [short_name(class_uri) for class_uri in model.classes_]
    class_report = {
        class_name: {
            key: float(value)
            for key, value in report[class_uri].items()
            if key in {"precision", "recall", "f1-score", "support"}
        }
        for class_uri, class_name in zip(model.classes_, class_tokens)
    }
    class_report["accuracy"] = float(report["accuracy"])

    ensure_directory(args.output_dir)
    (args.output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    (args.output_dir / "classification_report.json").write_text(json.dumps(class_report, indent=2), encoding="utf-8")
    (args.output_dir / "explanations.json").write_text(json.dumps(explanations, indent=2), encoding="utf-8")
    (args.output_dir / "explanation_evaluation.json").write_text(json.dumps(explanation_evaluation, indent=2), encoding="utf-8")

    print(json.dumps(metrics, indent=2))
    print(json.dumps(class_report, indent=2))
    print(json.dumps(explanation_evaluation, indent=2))


if __name__ == "__main__":
    main()