# Mini Project Report

## 1. Dataset

I used the AIFB RDF dataset, which is a classic benchmark for node classification on a heterogeneous knowledge graph.

The provided files are:

- `aifbfixed_complete.n3` for the RDF graph
- `trainingSet.tsv` for training labels
- `testSet.tsv` for test labels
- `completeDataset.tsv` for the full labeled set

The task is to predict the research group affiliation of a person from the RDF neighborhood around that person.

## 2. Method

My pipeline is intentionally simple and easy to reproduce:

1. Parse the RDF graph with `rdflib`
2. Build a 1-hop neighborhood feature vector for each person
3. Train a logistic regression classifier on the training split
4. Evaluate the classifier on the test split
5. Generate explanations from the learned model coefficients
6. Check explanation quality with a deletion-based faithfulness test

The main idea is that each person is represented by counts of RDF-related features such as:

- outgoing predicates
- incoming predicates
- RDF types
- predicate-object combinations

I chose this approach because it is easy to understand, it works directly on RDF data, and it produces explanations that can be inspected manually.

## 3. Predictive Performance

The validation run on the provided split produced the following results:

- Accuracy: 0.9444
- Macro F1: 0.9092
- Weighted F1: 0.9414
- Majority-class baseline accuracy: 0.4167

The model is clearly better than the baseline, which shows that the RDF neighborhood features contain useful information for the classification task.

## 4. Explanations

I generated local explanations by looking at the strongest positive feature contributions for the predicted class.

Example explanation files are written to `results/explanations.json`.

To evaluate explanations, I removed the most important features and checked how much the predicted confidence dropped.

Explanation evaluation results from the validation run:

- Average confidence drop: 0.7097
- Median confidence drop: 0.6741
- Prediction flip rate: 0.75

This suggests that the selected features are meaningful for the classifier, because removing them changes the predictions often and lowers confidence strongly.

## 5. Reproducibility

The project can be reproduced with these commands:

```bash
pip install -r requirements.txt
python run_experiment.py --data-dir . --output-dir results
```

The full implementation is in `run_experiment.py`.

## 6. Short Conclusion

This project solves the task with a straightforward RDF classification pipeline. It is not a copied paper reproduction: I used the AIFB benchmark, created my own 1-hop feature engineering, trained a simple classifier, and added a basic explanation evaluation step.