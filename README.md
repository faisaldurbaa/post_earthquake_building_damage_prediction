# Post-Earthquake Building Damage Prediction with Deep Learning

**AI model to predict post-earthquake building damage levels using structural characteristics and deep neural networks.**

📖 **[Read the full project article](https://faisaldurbaa.com/projects/earthquake-damage-prediction/)**

## 🎯 Objective
Support rescue teams with informed decision-making by predicting building damage severity (Grades 1-3) based on pre-earthquake structural features.

## 📊 Dataset
- **Source**: Nepal 2015 Earthquake Dataset
- **Size**: 260,601 buildings, 39 features
- **Target**: 3-class damage grades (Low/Medium/Severe)

## 🏗️ Model Architecture
- **Type**: Deep Neural Network (Sequential)
- **Structure**: 128 → Dropout(0.3) → 64 → Dropout(0.3) → 3 (Softmax)
- **Framework**: TensorFlow/Keras
- **Optimization**: Manual class weighting, early stopping

## 📈 Performance
- **Overall Accuracy**: 71%
- **Macro F1-Score**: 0.68
- **Class Balance**: Achieved through weighted training ({0:2.9, 1:1, 2:1.3})

| Grade | Precision | Recall | F1-Score |
|-------|-----------|--------|----------|
| 1 (Low) | 0.50 | 0.77 | 0.61 |
| 2 (Medium) | 0.78 | 0.71 | 0.74 |
| 3 (Severe) | 0.70 | 0.69 | 0.70 |

## 🔧 Tech Stack
- **ML/DL**: TensorFlow, Keras, scikit-learn
- **Data**: pandas, numpy
- **Visualization**: matplotlib

## 📁 Project Structure
```
Earthquake_Building_Damage_Prediction_DNN.ipynb
├── Phase 1: Data Collection & Literature Review
├── Phase 2: Exploratory Data Analysis
├── Phase 3: Preprocessing (scaling, encoding, splitting)
├── Phase 4: Model Development
├── Phase 5: Initial Testing & Evaluation
├── Phase 6: Optimization (weights, hyperparameters, structure)
└── Phase 7: Final Model & Results
```

## 🚀 Quick Start
1. **Install dependencies**:
   ```bash
   pip install tensorflow scikit-learn pandas numpy matplotlib
   ```

2. **Run the notebook**:
   ```bash
   jupyter notebook Earthquake_Building_Damage_Prediction_DNN.ipynb
   ```

3. **Key preprocessing steps**:
   - Feature selection (dropped low-impact features)
   - One-hot encoding for categorical variables
   - Standard scaling for numerical features
   - Stratified train/validation/test split (70/15/15)

## 🎯 Key Features
- **Comprehensive EDA** with feature importance analysis
- **Class imbalance handling** through weighted training
- **Robust preprocessing** pipeline preventing data leakage
- **Model optimization** across structure, hyperparameters, and weights
- **Geographic features** (geo_level_1_id, geo_level_2_id) for location-based patterns

## ⚠️ Limitations
- **Generalizability**: Limited to Nepal 2015 earthquake conditions
- **Data scope**: Single earthquake event dataset
- **Missing features**: Lacks seismic intensity measures (magnitude, acceleration)

## 🔍 Model Insights
- **Strongest predictors**: Construction materials, geographic location
- **Moderate predictors**: Building height, floor count, site conditions  
- **Weak predictors**: Age, family count, secondary use features
