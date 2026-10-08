# SupportSense — Model Card

## Model Summary

SupportSense is a supervised machine-learning classifier
that predicts banking customer-support intents.

**Selected model:** TF-IDF + Linear SVM

**Framework:** scikit-learn

**Dataset:** BANKING77

**Number of intent classes:** 77

## Intended Use

The model is intended for educational demonstrations
and experimentation with banking-support intent classification.

Potential applications include assisting with customer-support
ticket categorisation and routing.

It is not intended to make financial decisions or take
actions on customer accounts.

## Training Data

- Official training examples: 10,003
- Official test examples: 3,080
- Validation strategy: Stratified training/validation split
- Random seed: 42

The official test set was not used for hyperparameter tuning.

## Evaluation

The model was evaluated on the official BANKING77 test set.

- Accuracy: **0.8932**
- Macro Precision: **0.8967**
- Macro Recall: **0.8932**
- Macro F1: **0.8934**

## Model Selection

TF-IDF + Linear SVM was selected after comparison with
PyTorch and TensorFlow embedding classifiers.

Selection rationale:

The TF-IDF + Linear SVM was selected **as the production model because it achieved the highest test performance, with 89.32% accuracy and 89.34% macro F1, compared with 87.86% / 87.85% for TensorFlow/Keras and 83.31% / 83.35% for PyTorch. Its precision (89.67%) and recall (89.32%) were also well balanced. Given that the neural-network approaches introduced greater implementation complexity without improving performance, TF-IDF + Linear SVM provided the strongest balance of performance, simplicity, and practical deployability**.

## Known Limitations

- The model is specific to English banking-support language.
- It may misclassify ambiguous or out-of-domain messages.
- TF-IDF does not directly represent contextual word meaning.
- Raw Linear SVM decision scores are not calibrated probabilities.
- Model performance may change on real-world support traffic.

## Responsible Use

The application should not receive real banking credentials,
account numbers, or sensitive personal information.

Predictions should not be treated as guaranteed correct.