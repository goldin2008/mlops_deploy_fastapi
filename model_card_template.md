# Model Card
The predictive model is used to predict salaries using a classification model trained with logistic regression on publicly available Census Bureau data.

For additional information see the Model Card paper: https://arxiv.org/pdf/1810.03993.pdf

## Model Details
Used Random forest classifier for prediction. Default configuration were used for training.

## Intended Use
This model should be used to predict the category of the salary of a person.

## Training Data
Source of data https://archive.ics.uci.edu/ml/datasets/census+income ; 80% of the data is used for training using strtified KFold.


## Evaluation Data
Source of data https://archive.ics.uci.edu/ml/datasets/census+income ; 20% of the data is used to validate the model.


## Metrics
_Please include the metrics used and your model's performance on those metrics._
The model was evaluated using Accuracy score, F1 beta score, Precision and Recall. The value is around 0.72.


## Ethical Considerations
Data is open sourced on UCI machine learning repository for educational purposes. The UCI should be given proper credit and should be mentioned if model is publicly.


## Caveats and Recommendations
The data is biased based on gender. Have data imbalance that need to be investigated. The data had whitespaces leading to multiple false unique categories. So make sure to go through the data cleaning process to remove all those whitespaces.

