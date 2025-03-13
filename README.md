# Job Title Classification to Persona Segments

## Overview
This repository contains a machine learning system that classifies job titles into predefined Persona Segments. The system is built using Python and `scikit-learn` and is designed to be **simple and easy to use** via command-line execution.

## Persona Segments (Priority Order)
When classifying a job title, the system prioritizes assignments based on this order:
1. **GenAI**  
2. **Engineering**  
3. **Product**  
4. **Trust & Safety**  
5. **Legal & Compliance**  
6. **Executive**  

If a job title could fall into multiple categories, the highest-priority category is selected.

---

## Installation

### **1. Clone the Repository**
```sh
git clone <repo-url>
cd <repo-name>
```

### **2. Install Dependencies**
Ensure you have Python installed (version 3.8 or later). Then, install the required dependencies:
```sh
pip install -r requirements.txt
```

---

## Usage

### **1. Running Predictions**
To classify job titles, follow these steps:

#### **Step 1: Prepare the Input File (`data/input.csv`)**
Your input file must contain **two columns**:

| Column Name | Description |
|-------------|------------|
| `Record ID` | A unique identifier (e.g., from HubSpot) |
| `Job title` | The job title to be classified |

Example `data/input.csv`:
```csv
Record ID,Job title
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
| `Job title` | Same as input |
| `Persona Segment` | Assigned category |
| `Confidence Score` | Model confidence (rounded to the nearest multiple of 5) |

Example `tagged_personas.csv` output:
```csv
Record ID,Job title,Persona Segment,Confidence Score
37462838462827,AI Engineer,GenAI,90
82736482736473,Senior Software Developer,Engineering,85
ProductLead123,Product Lead,Product,80
```

---

### **2. Retraining the Model**

If the training data needs to be updated, you can retrain the model.

#### **Step 1: Prepare Training Data (`data/training_data.csv`)**
Your training data must contain **two columns**:

| Column Name | Description |
|-------------|------------|
| `Job title` | The job title text |
| `Persona Segment` | Correct category label |

Example `data/training_data.csv`:
```csv
Job title,Persona Segment
Sr. Product Manager,Product
Lead AI Researcher,GenAI
VP of Legal,Legal & Compliance
Senior Software Engineer,Engineering
Trust & Safety Specialist,Trust & Safety
CTO,Executive
```

#### **Step 2: Run Training**
Execute the following command:
```sh
make train
```

#### **Step 3: Confirmation**
If training succeeds, you will see:
```sh
✅ The model has been retrained
```
If there's an issue, you will see:
```sh
❌ There was an error retraining the model.
```

---

### **3. Using Title Standardization (`data/title_reference.csv`)**

To improve accuracy, you can provide a mapping file to **standardize job titles before classification**.

#### **Step 1: Prepare the Title Reference File (`data/title_reference.csv`)**
Your title reference file must contain **two columns**:

| Column Name | Description |
|-------------|------------|
| `Reference` | Shortened or alternative job title |
| `Standardization` | The corrected full job title |

Example `data/title_reference.csv`:
```csv
Reference,Standardization
Sr. PM,Senior Product Manager
ML Eng,Machine Learning Engineer
VP T&S,Vice President of Trust & Safety
CEO,Chief Executive Officer
Data Sci,Data Scientist
```

#### **Step 2: Update & Use the File**
- Place the updated `data/title_reference.csv` in the `data/` folder.
- The system will automatically apply standardization when running predictions.

---

## Project Structure
```
/project-root
│── model/                    # Stores trained model
│── data/                      # Contains input.csv, training_data.csv, title_reference.csv
│── scripts/                   # Scripts for training and prediction
│── logs/                      # Keeps logs of runs
│── tagged_personas.csv        # The output file
│── Makefile                   # Simplifies command execution
│── requirements.txt            # Dependencies
│── .gitignore                  # Best practices for version control
│── README.md                   # Detailed usage guide
```

---

## **Troubleshooting**

| Issue | Cause | Solution |
|--------|------------|------------|
| `❌ There was an error retraining the model.` | Training data may be missing or formatted incorrectly. | Ensure `data/training_data.csv` exists and has the correct format. |
| `❌ There was an error tagging the input file.` | Input file might be missing or malformed. | Check `data/input.csv` for missing columns or bad formatting. |
| Model outputs incorrect classifications. | Training data may need more examples. | Update `data/training_data.csv` and retrain using `make train`. |
| Confidence scores are too low. | Model may need better training data. | Add more diverse and representative job titles to `data/training_data.csv`. |

---

## **Conclusion**
This system provides an efficient way to classify job titles into predefined **Persona Segments** with **priority-based classification** and **confidence scoring**. By updating training data and title references, you can continuously improve its accuracy.

For any issues, refer to the **Troubleshooting** section or modify training data as needed.

---

This **README.md** now includes:
✅ **Detailed installation steps**  
✅ **Step-by-step user guide**  
✅ **File format descriptions for input, training, and title reference files**  
✅ **Project structure explanation**  
✅ **Troubleshooting section**  

Let me know if you need any refinements! 🚀
