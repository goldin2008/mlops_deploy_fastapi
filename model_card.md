
# Model Card

  
## Model Details
- This classifier trained to predict whether an employer's  income exceeds $50K/year.
- **Random Forest** and **Logistic Regression**, only **Random Forest** will be considered for evaluation since it shows better results.
- Developed by me for the third project for Udacity's Machine Learning DevOps nanodegree, October 2021.

<img src="screenshots/model_pipeline.PNG" width="400" height="200">

## Intended Use
- Intended to be used to determine what features impacts the income of a person.
- Intended to determine underprivileged employers.
- Not suitable for modern dates since the data is quite old.
  
## Factors
 - Evaluate on features that may be underprivileged such as gender, race, etc.
 
## Training Data
- Census Income [Dataset](https://archive.ics.uci.edu/ml/datasets/census+income) from UCI
- Categorical data:
  - Handled missing values by imputing the data using `SimpleImputer` with the most frequent value
  - Encoded the categories using `LabelEncoder` and setting a value of 1000 for unknown categories
- Numerical data:
  - Normalized the numerical data using `StandardScaler`
- Dropped the `education` column because it is already available encoded in the `education-num` column

## Evaluation Data
- Splitting the train data using sklearn `train_test_split` with a fixed `random_state=17` and stratified on `salary label`.
  
## Metrics
- Evaluation metrics includes **Precision**, **Recall** and **F1 score**.
- These 3 metrics can be calculated from the confusion matrix for binary classification which are more suitable for imbalanced problems.
- Precision: Ratio between correct predictions and the total predictions
- Recall: Ratio of the correct predictions and the total number of correct items in the set
- F1: Harmoinc mean between Precision and Recall to show the balance between them.

## Ethical Considerations
- Data is open sourced on UCI machine learning repository for educational purposes.

## Caveats and Recommendations
- The data was collected in 1996 which does not reflect insights from the modern world.
- Features with minor categories should be focused more when collecting extra data.

## Quantitative Analyses
All results shown are calculated for class 1 (>50K) using sklearn metrics
|				|Train |Test   |
|---------------|------|-------|
|Precision		|0.715 |0.688  |
|Recall         |0.903 |0.868  |
|F1          	|0.592 |0.570  |

<img src="plots/slice_metrics_sex_test.png" width="500" height="250">
<img src="plots/slice_metrics_race_test.png" width="500" height="250">
