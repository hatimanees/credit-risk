# Credit Risk Analysis

This project analyzes credit risk based on various factors, providing insights into which borrower characteristics correlate with an increased likelihood of default. The dataset used includes borrower financial and demographic information.

## Key Insights

1. **Interest Rate and Default Risk**: 
   - Borrowers with higher interest rates tend to default more frequently.
   - Average interest rate for defaulters: **13%**
   - Average interest rate for non-defaulters: **10.5%**

2. **Home Ownership and Default Risk**:
   - **Renters** are at a higher risk of default compared to other ownership statuses.
   - **Mortgage holders** show a relatively lower default risk.

3. **Loan Purpose and Default Contribution**:
   - Loans for **Debt Consolidation** and **Medical expenses** contribute more heavily to defaults.
   - In contrast, loans for **Personal use**, **Education**, and **Ventures** have a lower default contribution relative to their overall distribution.

4. **Employment Length and Default Risk**:
   - Borrowers with employment length **less than 6 years** are more likely to default.
   - Breakdown:
     - **70.5%** of non-defaulters have employment lengths under 6 years.
     - **29.5%** have employment lengths over 6 years.

5. **Loan Percent Income and Default Risk**:
   - The majority of non-defaulters (**94.3%**) have a loan-to-income ratio below **0.3**.
   - Higher loan-to-income ratios correlate with a higher likelihood of default.

6. **Income Level and Default Risk**:
   - Borrowers with annual income **above $30,000** are significantly less likely to default.
   - Breakdown:
     - **80.2%** of non-defaulters have an income above $30,000.
     - **19.8%** of non-defaulters have an income below $30,000.

7. **Maximum Loan Amount**:
   - The maximum loan amount in the dataset is **$35,000**.

## Repository Overview

- **Data**: The project uses historical data on borrower characteristics, loan details, and default status to assess credit risk.
- **Modeling**: Predictive models are developed to classify potential defaulters based on input features.
- **Evaluation**: Performance metrics assess the model's ability to correctly predict default status.

## Getting Started

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/hatimanees/credit-risk.git
   cd credit-risk
   ```

2. **Dependencies**: Install the required packages using:
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the Analysis**: 
   The analysis can be run in a Jupyter notebook or a Python script. Follow the instructions in the code to explore different features and insights.

## Model Usage

The repository includes a machine learning model to predict default status based on borrower characteristics. The model outputs:
- **0**: Borrower will not default.
- **1**: Borrower is likely to default.

## Docker Support

To run the project in a Docker container:
1. Pull the Docker image:
   ```bash
   docker pull hatimanees/credit-risk
   ```

2. Run the Docker container:
   ```bash
   docker run -p 8080:8080 hatimanees/credit-risk
   ```

## Additional Resources

- **Project Video**: [Project Explanation](https://drive.google.com/file/d/1nDHLevXd9bF3vZ7_2RckxUpyhbVXHGT0/view?usp=sharing)
- **GitHub Repository**: [Credit Risk GitHub Repo](https://github.com/hatimanees/credit-risk)

---

