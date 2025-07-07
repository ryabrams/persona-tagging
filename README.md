# 🔎 Persona Classification

## 📝 Overview
This repository contains a machine learning system that classifies job titles into predefined Persona Segments. The system combines keyword-based rules with a machine learning model built using Python and `scikit-learn`.

## 🪣 Persona Segments
When classifying a job title, the system prioritizes assignments based on this order:
1. **GenAI**  
2. **Engineering**  
3. **Product**  
4. **Cyber Security**
5. **Trust & Safety**  
6. **Legal & Compliance**  
7. **Executive**  

If a job title could fall into multiple categories, the highest-priority category is selected.

---

## 🛠️ Installation

### **1. Clone the Repository**
```sh
git clone <repo-url>
cd <repo-name>
```

### **2. Install Dependencies**
Ensure you have Python installed (version 3.7 or later). Then, install the required dependencies:
```sh
pip install -r requirements.txt
```

### **3. Verify Installation**
Check that all required files and directories are in place:
```sh
make check
```

---

## ♻️ Usage

### **Quick Start**
```sh
# Run the complete pipeline (train + predict)
make all

# Or run steps individually:
make train    # Train the model
make predict  # Run predictions
```

### **1. Running Predictions**
To classify job titles, follow these steps:

#### **Step 1: Prepare the Input File (`data/input.csv`)**
Your input file must contain **two columns** (case-insensitive):

| Column Name | Description |
|-------------|------------|
| `Record ID` | A unique identifier (e.g., from HubSpot) |
| `Job Title` | The job title to be classified |

Example `data/input.csv`:
```csv
Record ID,Job Title
37462838462827,AI Engineer
82736482736473,Senior Software Developer
ProductLead123,Product Lead
```

#### **Step 2: Run Prediction**
Execute the following command:
```sh
make predict
```

#### **Step 3: View Results (`tagged_personas.csv`)**
The output file will be `tagged_personas.csv`, containing:

| Column Name | Description |
|-------------|------------|
| `Record ID` | Same as input |
| `Job Title` | Standardized job title (if applicable) |
| `Persona Segment` | Assigned category (empty if confidence < 50%) |
| `Confidence Score` | Model confidence (0-100, rounded to nearest 5) |

Example `tagged_personas.csv` output:
```csv
Record ID,Job Title,Persona Segment,Confidence Score
37462838462827,AI Engineer,GenAI,90
82736482736473,Senior Software Developer,Engineering,85
ProductLead123,Product Lead,Product,80
```

---

### **2. Training the Model**

#### **Step 1: Prepare Training Data (`data/training_data.csv`)**
Your training data must contain **two columns** (case-insensitive):

| Column Name | Description |
|-------------|------------|
| `Job Title` | The job title text |
| `Persona Segment` | Correct category label |

Example `data/training_data.csv`:
```csv
Job Title,Persona Segment
Sr. Product Manager,Product
Lead AI Researcher,GenAI
VP of Legal,Legal & Compliance
Senior Software Engineer,Engineering
Trust & Safety Specialist,Trust & Safety
Security Architect,Cyber Security
CTO,Executive
```

#### **Step 2: Run Training**
Execute the following command:
```sh
make train
```

#### **Step 3: Review Training Output**
The training process now provides detailed metrics:
```
=== Data Quality Report ===
Persona Segment Distribution:
  Engineering: 523 samples (32.1%)
  Product: 387 samples (23.8%)
  GenAI: 201 samples (12.3%)
  ...

=== Model Evaluation ===
Classification Report:
              precision    recall  f1-score   support
Engineering       0.92      0.89      0.90       105
Product          0.88      0.91      0.89        78
...

Cross-validation scores (5-fold):
  Mean accuracy: 0.876 (+/- 0.042)
  Test set accuracy: 0.883

✅ Model training completed successfully!
```

The system also saves metadata to `model/model_metadata.txt` with training details.

---

### **3. Optional: Keyword-Based Rules (`data/keyword_matching.csv`)**

For precise control, you can define keyword rules that take priority over ML predictions.

#### **File Format**
| Column Name | Description |
|-------------|------------|
| `Keyword` | Text to match (case-insensitive) |
| `Rule` | Either `contains` or `equals` |
| `Persona Segment` | Segment to assign |
| `Exclude Keyword` | Optional: exclude if this text is present |

Example `data/keyword_matching.csv`:
```csv
Keyword,Rule,Persona Segment,Exclude Keyword
chief executive,contains,Executive,
ai,contains,GenAI,
engineer,contains,Engineering,sales
product manager,equals,Product,
```

**Note**: Keyword matches receive 100% confidence and override ML predictions.

---

### **4. Optional: Title Standardization (`data/title_reference.csv`)**

Standardize job title variations before classification.

#### **File Format**
| Column Name | Description |
|-------------|------------|
| `Reference` | Original/variant job title |
| `Standardization` | Standardized form |

Example `data/title_reference.csv`:
```csv
Reference,Standardization
Sr. PM,Senior Product Manager
ML Eng,Machine Learning Engineer
VP T&S,Vice President of Trust & Safety
CEO,Chief Executive Officer
eng,Engineer
```

**Note**: Standardization is case-insensitive and applied during both training and prediction.

---

## 🛠️ Configuration

### **Classification Process**
1. **Keyword Matching** (if `keyword_matching.csv` exists)
   - Applied first, receives 100% confidence
   - Case-insensitive matching
   
2. **Title Standardization** (if `title_reference.csv` exists)
   - Applied to remaining titles
   - Case-insensitive lookup
   
3. **ML Classification**
   - Uses TF-IDF features with n-grams (1-3)
   - Logistic Regression with balanced class weights
   - Priority enforcement for uncertain predictions

### **Thresholds**
- **Confidence Threshold**: 50% (predictions below this are left unassigned)
- **Priority Enforcement**: Applied when model confidence < 70%
- **Fuzzy Matching**: Available but disabled by default

### **Available Commands**
```sh
make help     # Show all available commands
make train    # Train the model
make predict  # Run predictions
make all      # Run full pipeline (train + predict)
make check    # Verify system setup
make validate # Validate format of input files
make clean    # Remove generated files
make retrain  # Force model retraining
make test     # Quick test using training data
```

---

## 🧩 Project Structure
```
/project-root
├── model/                    
│   ├── persona_classifier.pkl      # Trained model (generated)
│   └── model_metadata.txt         # Training metadata (generated)
├── data/
│   ├── input.csv                  # Input file for predictions
│   ├── training_data.csv          # Training data
│   ├── keyword_matching.csv       # Optional: keyword rules
│   └── title_reference.csv        # Optional: title standardization
├── scripts/
│   ├── predict.py                 # Prediction script
│   ├── train_model.py            # Training script
│   └── title_standardizer.py     # Standardization module
├── tagged_personas.csv            # Output file (generated)
├── Makefile                       # Build automation
├── requirements.txt               # Python dependencies
├── README.md                      # This file
└── LICENSE                        # License file
```

---

## 📊 Model Performance

The system provides comprehensive performance metrics during training:
- **Classification Report**: Precision, recall, and F1-score per persona
- **Cross-validation**: 5-fold CV with mean accuracy and standard deviation
- **Test Set Accuracy**: Hold-out test performance
- **Data Quality Checks**: Class distribution, imbalance warnings, duplicate detection

---

## 😱 Troubleshooting

| Issue | Cause | Solution |
|-------|-------|----------|
| `❌ Model file not found` | Model hasn't been trained | Run `make train` first |
| `❌ Training file not found` | Missing training data | Ensure `data/training_data.csv` exists |
| `❌ Input file not found` | Missing input data | Create `data/input.csv` with required columns |
| `Invalid persona segments in training data` | Typo in persona names | Check spelling matches valid personas exactly |
| `Found X rows with duplicate Record IDs` | Non-unique identifiers | Review and fix duplicate IDs in input file |
| Low confidence scores | Insufficient training data | Add more diverse examples to training data |
| Wrong classifications | Model needs retraining | Update training data and run `make retrain` |
| `High class imbalance detected` | Uneven persona distribution | Add more samples for underrepresented personas |

### **Data Validation**
Run `make validate` to check your input files for common issues:
- File existence and readability
- Column names and counts
- Number of rows in each file
- Basic data format validation

### **Logging**
The system provides detailed logging during execution:
- **INFO**: Normal operations and statistics
- **WARNING**: Potential issues (e.g., missing files, imbalanced data)
- **ERROR**: Fatal errors that stop execution

---

## 🚀 Advanced Features

### **Fuzzy Matching** (Experimental)
Enable approximate title matching in `title_standardizer.py`:
```python
standardized_title = standardize_title(title, use_fuzzy=True, fuzzy_threshold=0.8)
```

### **Model Metadata**
After training, check `model/model_metadata.txt` for:
- Training date and time
- Dataset statistics
- Model parameters
- Performance metrics

---

## 🪪 License
This project is licensed under the MIT License.