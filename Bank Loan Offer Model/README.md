# datascienceportfolio
# Predict Growth of Bank Asset Customers

## Project Overview
This project aims to predict which deposit-only customers at a bank can be converted into asset customers (i.e., those who take out loans). The ability to predict customer behavior allows the bank to focus its marketing efforts more effectively, increasing revenue by targeting customers likely to take out loans.

## Problem Statement
Banks rely on loans and other asset products for significant revenue. This project develops predictive models to identify deposit-only customers who are likely to take out loans. By using machine learning, the bank can improve marketing campaigns and increase conversions to asset customers.

## Data Source
The dataset used for this project was sourced from [Kaggle](https://www.kaggle.com/datasets/itsmesunil/bank-loan-modelling), consisting of 5,000 records of bank customers. Key features include:
- Income
- Average credit card usage
- Education level
- Other demographic factors

## Methodology

### Data Exploration
Exploratory data analysis (EDA) was performed to understand the key factors influencing a customer's likelihood to take out a loan. Visualizations such as correlation heatmaps, histograms, and boxplots were used to identify patterns.

### Modeling
Two models were developed and compared:
- **Linear Regression:** A simple model used to interpret basic relationships between features and loan acceptance.
- **Gradient Boosting:** A more complex model that captures non-linear relationships and interactions between features.

### Evaluation Metrics
The models were evaluated using several metrics:
- **Accuracy:** Measures how often the model made correct predictions overall.
- **Precision:** Indicates the proportion of true positive predictions out of all positive predictions.
- **Recall:** Represents the proportion of true positives that were correctly identified.
- **F1 Score:** A harmonic mean of precision and recall, providing a balanced measure of performance.

#### Model Results
- **Linear Regression Results:**
  - Accuracy: 0.98
  - Precision: 0.95
  - Recall: 0.80
  - F1 Score: 0.87
- **Gradient Boosting Results:**
  - Accuracy: 0.99
  - Precision: 0.94
  - Recall: 0.90
  - F1 Score: 0.92

The Gradient Boosting model outperformed Linear Regression, particularly in recall and F1 score, making it the preferred model for predicting loan acceptance.

## Feature Importance
The most important features for predicting loan acceptance were:
- Income
- Average monthly credit card usage (CCAvg)
- Education level
- Family size

## Recommendations
The bank should focus its marketing efforts on customers with higher income, significant credit card usage, and advanced education, as they are most likely to convert into asset customers. The Gradient Boosting model can be deployed for targeted marketing.

## Future Work
- Explore additional machine learning models (e.g., Support Vector Machines).
- Fine-tune the existing Gradient Boosting model.
- Experiment with new feature engineering techniques to improve predictive performance.

## Ethical Considerations
- Ensure that the model does not discriminate based on sensitive attributes such as age or income.
- Regularly audit the model for bias and fairness, and maintain transparency in decision-making.

## References
- Friedman, J. H. (2001). Greedy function approximation: A gradient boosting machine. *Annals of Statistics, 29*(5), 1189-1232.
- Piatetsky-Shapiro, G. (2014). *Applied Predictive Analytics: Principles and Techniques for the Professional Data Analyst*. Wiley.
- Sunil. (n.d.). Bank Loan Modelling. Kaggle. Retrieved from https://www.kaggle.com/datasets/itsmesunil/bank-loan-modelling
