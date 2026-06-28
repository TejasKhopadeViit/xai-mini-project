# AIFB RDF Classification and XAI Project

This project uses the AIFB RDF dataset to predict the research group affiliation of people from their RDF neighborhood.

The implementation is intentionally simple and student-friendly:

- RDF triples are parsed from `aifbfixed_complete.n3`
- Each person is represented by a small 1-hop neighborhood feature vector
- A logistic regression classifier is trained on the provided train split
- Explanations are generated from the model coefficients
- Explanation quality is checked with a simple deletion test

## Dataset

The dataset folder contains these files:

- `aifbfixed_complete.n3` - full RDF graph in Notation3 format
- `trainingSet.tsv` - training labels
- `testSet.tsv` - test labels
- `completeDataset.tsv` - combined label file

## Setup

Use Python 3.14 in the provided environment, or create a fresh virtual environment with a similar Python version.

Install the dependencies:

```bash
pip install -r requirements.txt
```

## Reproduce the Results

Run the full pipeline from the project root:

```bash
python run_experiment.py --data-dir . --output-dir results
```

This command will:

1. Parse the RDF graph
2. Build features for the train and test persons
3. Train the classifier
4. Evaluate predictive performance
5. Create instance explanations
6. Evaluate the explanations with a deletion-based faithfulness test

## Output Files

The script writes the following files into `results/`:

- `metrics.json` - accuracy and F1 scores
- `classification_report.json` - per-class precision/recall/F1
- `explanations.json` - example explanations for a few test instances
- `explanation_evaluation.json` - faithfulness metrics for the explanations

## Why This Approach

I used a linear model instead of a more complex GNN because the goal of the mini project is to show a complete and understandable RDF machine learning pipeline. The model is easy to run, easy to explain, and still suitable for a solid bachelor/master-level baseline.

## Notes

If you want to inspect or discuss the results in the report, first run the pipeline so the JSON files in `results/` are generated.