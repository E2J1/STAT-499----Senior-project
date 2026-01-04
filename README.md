# ⚽ Predicting Football Player Success Using Machine Learning on Football Manager 2024

This project was completed as part of the **Senior Project (STAT 499)** for the Statistics and Data Science program at the **University of Bahrain**. It uses Football Manager 2024 as a controlled simulation environment to predict long-term player success through machine learning.

## 📄 Full Report

The complete academic report (138 pages) is included in this repository:
- **Main Content** (Pages 11-60): Introduction through Conclusion covering objectives, methodology, results, discussion, and conclusions
- **Appendices** (Pages 65-138): Detailed data descriptions, hyperparameter search spaces, comprehensive visualizations (confusion matrices, ROC curves, PR curves, threshold plots), performance tables, feature importance plots, and cluster analysis figures

All experimental details, statistical tests, and model configurations are thoroughly documented in the report.

## 👥 Author

**Ebrahim Juma Shakak Alsawan**  
ID: 202009241  
**Supervisor:** Ms. Aseel Masoud Ebrahim Alhermi  
**Department of Mathematics, University of Bahrain**  
**December 2025**

---

## 🎯 Objective

To predict the future success of young football players (aged 15–23) by:

1. Simulating their 10-year career development in Football Manager 2024 **three times independently**
2. Extracting player attributes at Year-0 (start) and Year-10 (end)
3. Building machine learning models to classify players as **successful** or **unsuccessful**

**Success is defined through a dual-benchmark approach:**
- Real-world benchmark: **Top 25%** market value from Transfermarkt (500 most valuable players globally)
- In-game benchmark: **Top 10%** market value in FM after 10 years
- A player is labeled successful if they reach the top 10% threshold in **at least 2 out of 3 simulations**

---

## 📊 Dataset Overview

| Feature             | Description                                                                 |
|---------------------|-----------------------------------------------------------------------------|
| **Source**          | Football Manager 2024 simulation + Transfermarkt 2025 real-world data      |
| **Players**         | 43,094 (filtered from ~88,000 for ages 15–23, top 18 leagues)              |
| **Simulation**      | 10 years × 3 independent runs per player                                    |
| **Attributes**      | Technical, Mental, Physical, Hidden (CA, PA), Demographics, Market Value    |
| **Target**          | `success_label` (1 = successful in ≥2/3 simulations, 0 = unsuccessful)     |
| **Class Balance**   | ~3.3% successful (extreme imbalance)                                        |

### Included Leagues (Top 18)
Argentina, Belgium, Brazil, Croatia, Denmark, England (Premier League + Championship), France, Germany (Bundesliga + 2. Bundesliga), Italy, Japan, Mexico, Netherlands, Poland, Portugal, Spain, United States (MLS)

---

## 🔍 Key Findings

### Model Performance Summary

**Best Overall Model: XGBoost (Full Mode, With Age)**

| Threshold Type           | Balanced Accuracy | Precision | Recall | F1-Score | MCC    |
|--------------------------|-------------------|-----------|--------|----------|--------|
| **Balanced Accuracy**    | **0.8999**        | 0.1642    | 0.9677 | 0.2807   | 0.3609 |
| **F1-Optimized**         | 0.7542            | 0.4113    | 0.5346 | **0.4649** | 0.4484 |

**Top 3 Models (Balanced Accuracy Ranking):**
1. XGBoost (Full – With Age): 0.8999
2. Random Forest (Full – With Age): 0.8872
3. Logistic Regression (Full – With Age): 0.8810

### Model Configurations Tested

The study employed a **four-way experimental design**:

**Configurations:**
- **With-Age vs. No-Age** – Isolates the contribution of age as a predictive feature
- **Realistic Mode vs. Full Mode** – Distinguishes between scout-visible attributes vs. complete game data (including hidden CA/PA)

This creates 4 combinations × 5 algorithms = **20 total model variants**

### Key Insights

1. **Hidden Attributes Dominate**: When CA (Current Ability) and PA (Potential Ability) are available, they overwhelmingly determine success
2. **Age Matters**: Including age consistently improved model performance by 4–7 percentage points in balanced accuracy
3. **Most Important Visible Attributes**:
   - **Mental**: Anticipation, Decisions, Determination, Composure, Concentration, Bravery
   - **Physical**: Strength, Balance, Pace, Stamina, Natural Fitness
   - **Technical**: First Touch, Technique, Passing
4. **Ensemble Methods Win**: XGBoost and Random Forest significantly outperformed linear models and single decision trees
5. **Extreme Imbalance Challenge**: Only 3.3% of players achieve elite success, making this an inherently difficult prediction task

---

## 🛠 Technology Stack

### Simulation & Data Extraction
- **Football Manager 2024**: Player career simulation environment
- **PyAutoGUI**: Automated data extraction from FM interface
- **Python**: Core programming language

### Data Processing & Analysis
- **pandas, NumPy**: Data manipulation and numerical operations
- **Scikit-learn**: Machine learning algorithms and preprocessing
- **XGBoost, LightGBM**: Gradient boosting models
- **imbalanced-learn (SMOTE)**: Class imbalance handling
- **Optuna**: Hyperparameter optimization (TPE sampler, 100 trials)

### Model Interpretability
- **SHAP**: Feature importance and model explanation
- **Permutation Importance**: Feature contribution analysis
- **Coefficient Analysis**: Linear model interpretation

### Visualization
- **Matplotlib, Seaborn**: Statistical visualizations
- **t-SNE, PCA**: Dimensionality reduction for cluster visualization

---

## 📈 Methodology Overview

### 1. Data Collection & Simulation
- Extracted Year-0 attributes for 43,094 players aged 15–23 from top 18 leagues
- Ran **three independent 10-year simulations** for each player
- Extracted Year-10 data to compute market values and success labels
- Used majority-vote rule: successful if top 10% in ≥2/3 simulations

### 2. Data Preprocessing
- Removed seasonal statistics and club-dependent attributes (wages, contracts)
- Handled "Not for Sale" values using Asking Price (AP) as technical solution
- Excluded CA/PA for Realistic Mode; included for Full Mode
- Applied stratified train-validation-test split (70%-15%-15%)

### 3. Class Imbalance Handling
- Applied **SMOTE** (Synthetic Minority Over-sampling Technique)
- Tested three oversampling ratios: 0.2, 0.5, 1.0
- Selected optimal ratio per model via Optuna

### 4. Model Training & Optimization
**Algorithms Evaluated:**
- Logistic Regression (baseline)
- Decision Tree
- Random Forest
- Support Vector Machine (RBF kernel)
- XGBoost

**Optimization Strategy:**
- Optuna with TPE sampler (100 trials per configuration)
- Primary metric: **Balanced Accuracy**
- Secondary analysis: F1-optimized threshold

### 5. Evaluation Framework
**Metrics Used:**
- Balanced Accuracy (primary)
- Precision, Recall, F1-Score
- ROC-AUC, Precision-Recall AUC
- Matthews Correlation Coefficient (MCC)
- Cohen's Kappa
- Geometric Mean (G-Mean)

**Dual-Threshold Evaluation:**
- Balanced-Accuracy-optimized threshold
- F1-optimized threshold

### 6. Interpretability Analysis
- SHAP values (XGBoost)
- Feature importance (Random Forest, Decision Tree, XGBoost)
- Coefficient analysis (Logistic Regression)
- Permutation importance (SVM)

### 7. Exploratory Cluster Analysis
- K-Means clustering on visible attributes and CA-PA space
- PCA and t-SNE visualizations
- Validation that CA/PA space cleanly separates successful players

---

## 📁 Repository Structure

```
├── videos/                                    # Demonstration videos
│   ├── 1_DEMO_Pyautogui.mp4                  # PyAutoGUI automation demo
│   ├── 2_data_merge_for_the_shortlist.mp4    # Data merging process
│   ├── 3_shortlist_extraction.mp4            # Shortlist extraction demo
│   └── 4_year_10_data_extraction.mp4         # Year-10 data extraction
│
├── notebooks/                                 # Jupyter notebooks
│   ├── pyautogui.ipynb                       # Data extraction automation
│   ├── shortlist script.ipynb                # Shortlist management
│   ├── year_10_data_extraction_script.ipynb  # Year-10 extraction
│   ├── Correlation.ipynb                     # Statistical analysis
│   │
│   ├── final_LR_With_Age.ipynb              # Logistic Regression (With Age)
│   ├── final_LR_Without_Age.ipynb           # Logistic Regression (No Age)
│   ├── final_RF_With_Age.ipynb              # Random Forest (With Age)
│   ├── final_RF_Without_Age.ipynb           # Random Forest (No Age)
│   ├── final_DT_With_Age.ipynb              # Decision Tree (With Age)
│   ├── final_DT_Without_Age.ipynb           # Decision Tree (No Age)
│   ├── final_SVC_With_Age.ipynb             # SVM (With Age)
│   ├── final_SVC_Without_Age.ipynb          # SVM (No Age)
│   ├── final_XGBoost_With_Age.ipynb         # XGBoost (With Age)
│   ├── final_XGBoost_Without_Age.ipynb      # XGBoost (No Age)
│   └── final_clusters_Without_Age.ipynb      # Cluster analysis
│
├── STAT 499 FINAL REPORT.pdf                 # Full academic report (138 pages: 50 main + 88 appendix)
├── LICENSE                                    # MIT License
└── README.md                                  # This file
```

---

## 🚀 Getting Started

### Prerequisites

```bash
pip install pandas numpy scikit-learn xgboost lightgbm imbalanced-learn
pip install shap optuna matplotlib seaborn pyautogui
```

**Note**: You need a legitimate copy of **Football Manager 2024** to replicate the data extraction process.

### Replicating the Study

1. **Extract Data from Football Manager**
   - Follow the PyAutoGUI automation scripts in the notebooks
   - Videos demonstrate the extraction process
   - Extract Year-0 data for all players in top 18 leagues (ages 15–23)

2. **Run Three 10-Year Simulations**
   - Simulate 10 in-game years
   - Reload to Year-0 save and repeat 2 more times
   - Extract Year-10 data after each simulation

3. **Preprocess Data**
   - Merge simulation runs using player UID
   - Apply majority-vote labeling (success in ≥2/3 runs)
   - Create Realistic (no CA/PA) and Full datasets

4. **Train Models**
   - Open the relevant notebook for each model configuration
   - Models use Optuna for hyperparameter optimization
   - Training includes SMOTE, StandardScaler, and full pipeline

5. **Evaluate & Interpret**
   - All evaluation metrics computed automatically
   - SHAP analysis and feature importance included in notebooks
   - Confusion matrices and curves generated

---

## 📊 Statistical Validation

### Age Analysis

**Point-Biserial Correlation**: r = 0.1213, p < 1.44 × 10⁻¹⁴³  
**Mann-Whitney U Test**: U = 42,866,537, p < 1.64 × 10⁻¹⁴⁹  
**Chi-Square Test**: χ² = 1237.84, p < 3.64 × 10⁻²⁷¹

**Success Rates by Age Group:**
- Young (≤20): 1.12% success rate
- Peak (21–23): 7.42% success rate

This confirms age provides genuine predictive value, not just proxy effects.

---

## 🔮 Limitations & Future Work

### Limitations
1. FM simplifies real-world psychological and environmental factors
2. Market value is a proxy, not a perfect measure of success
3. Only Year-0 and Year-10 captured (no mid-career tracking)
4. Dataset limited to top 18 leagues and ages 15–23
5. Hidden attributes (CA/PA) unavailable to real scouts

### Future Directions
1. **Time-series modeling**: Extract Year-1, Year-5 to capture development curves
2. **Deep learning**: LSTM/Transformer models for temporal patterns
3. **Expand coverage**: Include lower leagues and broader age ranges
4. **Contextual variables**: Incorporate injuries, coaching quality, playing time
5. **Multi-dimensional success**: Combine market value with international caps, trophies
6. **Cross-version validation**: Test on FM23, FM25 for robustness

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

Special thanks to:
- **Ms. Aseel Masoud Ebrahim Alhermi** for continuous supervision, patience, and invaluable guidance throughout this project
- **Department of Mathematics, University of Bahrain** for supporting this unconventional research topic and encouraging academic creativity
- **My family and friends** for their constant support, encouragement, and understanding throughout my university journey. Their motivation played a significant role in completing both this project and my degree
- **Sports Interactive** for creating Football Manager 2024, which made this research possible
- **Transfermarkt** for real-world valuation data

---

## 📧 Contact

For questions or collaboration:
- **Author**: Ebrahim Juma Shakak Alsawan
- **GitHub**: [E2J1](https://github.com/E2J1)

---

**Note**: Raw FM database files cannot be shared due to licensing restrictions, but all code and methodology are provided for full reproducibility by users with legitimate FM24 access.
