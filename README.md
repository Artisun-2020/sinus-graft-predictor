# 🦷 Sinus Graft Absorption Predictor

**AI-Powered Clinical Decision Support System for Maxillary Sinus Lift Surgery**

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://share.streamlit.io/)

## Overview

This web application predicts bone graft absorption volume (ΔAV) at 6-8 months post-surgery for maxillary sinus lift procedures. The prediction model is based on clinical data from 178 patients at Shandong University Stomatological Hospital.

## Features

### 📋 Model Introduction Module
- Training data information and statistics
- Variable ranges and value domains
- Model performance metrics (R², AUC, RMSE)
- ROC curve and DCA analysis
- Decision threshold recommendations

### 🔬 Individual Prediction Module
- Patient information input
- Real-time prediction
- Risk assessment (Low/High absorption risk)
- Clinical recommendations
- SHAP-based model interpretation

### 📊 External Validation & Batch Prediction Module
- Upload local datasets for validation
- Calculate performance metrics on local data
- Batch prediction capabilities
- Download prediction results

### ⚙️ Data Adjustment Module
- Column name mapping
- Variable format standardization
- Data preprocessing tools

## Model Performance

| Metric | Value |
|--------|-------|
| R² Score | 0.493 |
| AUC (Binary) | 0.814 |
| RMSE | 142.3 mm³ |
| Sensitivity | 0.78 |
| Specificity | 0.72 |

## Selected Features (LASSO)

1. **T1_AV** - Initial graft volume (most important)
2. **Age** - Patient age
3. **HbA1c** - Glycated hemoglobin level
4. **Membrane Status** - Schneider membrane perforation
5. **Smoking** - Smoking history

## How to Use

### Option 1: Streamlit Cloud (Recommended)
1. Fork this repository to your GitHub account
2. Go to [Streamlit Cloud](https://share.streamlit.io/)
3. Sign in with your GitHub account
4. Click "New app" and select this repository
5. Set the main file path to `app.py`
6. Click "Deploy"

### Option 2: Local Installation
```bash
# Clone the repository
git clone https://github.com/YOUR_USERNAME/sinus-graft-predictor.git
cd sinus-graft-predictor

# Install dependencies
pip install -r requirements.txt

# Run the app
streamlit run app.py
```

## Citation

If you use this tool in your research, please cite:

```
Yang Chen Lab. (2025). Sinus Graft Absorption Predictor: An AI-Powered Clinical 
Decision Support System. School of Stomatology, Shandong University.
```

## Disclaimer

This tool is for research and educational purposes only. Clinical decisions should always be made by qualified healthcare professionals based on comprehensive patient evaluation.

## Contact

- **Email**: yangchen@sdu.edu.cn
- **Institution**: School of Stomatology, Shandong University

## License

© 2025 Yang Chen Lab. All Rights Reserved.
