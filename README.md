# flightBookingprediction

# Flight Customer Propensity Prediction (Logistic Regression)

## Business Problem
A travel platform wants to know which users are likely to take its product (`Taken_product`), so marketing can focus on the right customers. This project builds a classification model to predict that.

## Dataset
Flight customer data with 11,760 rows and 17 columns (user activity, device, location preference, family size, company page engagement). After cleaning and removing 1,139 duplicates: 10,621 rows. [Add the source, e.g. Great Learning course dataset.]

## Approach
1. **Data cleaning:** fixed inconsistent labels (e.g. "Three" to 3, "*" to missing, 1/0 to Yes/No), grouped 11 device labels into Mobile and Laptop, merged location labels, fixed `Adult_flag` values to 0/1, removed duplicates
2. **EDA:** univariate, bivariate and multivariate analysis
3. **Preprocessing:** missing value treatment, outlier capping (IQR method), dummy encoding
4. **Modelling:** separate logistic regression models (statsmodels) for Mobile and Laptop users, using a stratified 70/30 train-test split
5. **Model refinement:** checked multicollinearity with VIF and removed insignificant features one at a time, checking that test performance held
6. **Evaluation:** accuracy, precision, recall, F1, ROC-AUC, confusion matrices and ROC curves on test data
7. **Interpretation:** converted coefficients into odds ratios

## Results (test data, final models)
| Metric | Mobile | Laptop |
|---|---|---|
| Accuracy | 0.87 | 0.82 |
| Precision | 0.70 | 0.73 |
| Recall | 0.21 | 0.46 |
| F1 | 0.32 | 0.56 |
| ROC-AUC | 0.78 | 0.84 |

## Key Insights
- Following the company page is the strongest signal: it raises the odds of taking the product about 5x (Mobile) and 6x (Laptop)
- Adult_flag = 1 lowers the odds by about 72% (Mobile) and 63% (Laptop)
- A higher travelling network rating lowers the odds, while more weeks since the last outstation check-in raises them
- [Add 1-2 insights from your EDA]

## Limitations and Next Steps
- Recall is low, especially for Mobile (0.21), because the classes are imbalanced and the default 0.5 threshold is used. Next steps: tune the threshold, try class weights, and test tree-based models
- Report metric choice: [add your rationale, e.g. why you prioritised recall or F1]

## Tools
Python, pandas, NumPy, matplotlib, seaborn, statsmodels, scikit-learn

## Files
- `VishalUpadhyaya_flightBokingPrediction_Notebook.ipynb`: full notebook
