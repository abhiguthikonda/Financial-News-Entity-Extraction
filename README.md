# Financial Named Entity Recognition (NER) System

A machine learning system for extracting key financial entities from news articles and reports.

## 📋 Project Description

The NER system for financial text analysis streamlines the extraction of key entities such as company names, stock tickers, and financial metrics from news articles and reports. By automating this process, the system enhances data accessibility and accuracy, allowing for quicker analysis and timely reporting. This technology improves the efficiency and effectiveness of financial news agencies and market analysts.

## 🏷️ Entity Types

| Entity | Description | Examples |
|--------|-------------|----------|
| **COMPANY** | Company names | Tesla, Apple, Microsoft |
| **TICKER** | Stock symbols | AAPL, TSLA, MSFT |
| **CURRENCY** | Currency mentions | $, USD, EUR |
| **INDICATOR** | Financial metrics | revenue, profit, EPS |
| **EVENT** | Financial events | IPO, merger, acquisition |

## 📁 Project Structure

```
financial_ner_project/
├── train_ner_models.ipynb    # Jupyter notebook for training models
├── requirements.txt          # Python dependencies
├── data/                     # Training data and visualizations
├── models/                   # Saved model files
│   ├── best_ner_model.pkl
│   ├── tfidf_vectorizer.pkl
│   ├── label_mapping.json
│   ├── model_metadata.json
│   └── financial_ner_model.zip
├── backend/                  # FastAPI backend
│   ├── main.py
│   └── requirements.txt
└── frontend/                 # Web interface
    ├── index.html
    ├── styles.css
    └── app.js
```

## 🚀 Quick Start

### 1. Install Dependencies

```bash
# Create virtual environment (optional but recommended)
python -m venv venv
venv\Scripts\activate  # Windows
# source venv/bin/activate  # Linux/Mac

# Install training dependencies
pip install -r requirements.txt

# Install backend dependencies
pip install -r backend/requirements.txt
```

### 2. Train Models

Open and run the Jupyter notebook:

```bash
jupyter notebook train_ner_models.ipynb
```

Run all cells to:
- Train 7 different ML models
- Compare their performance
- Save the best model
- Create a ZIP archive

### 3. Start the Backend

```bash
cd backend
python main.py
```

The API will be available at:
- API: http://localhost:8000
- Docs: http://localhost:8000/docs

### 4. Open the Frontend

Simply open `frontend/index.html` in your browser, or serve it:

```bash
cd frontend
python -m http.server 3000
```

Then visit http://localhost:3000

## 📊 Models Compared

The notebook trains and compares these models:

1. **Logistic Regression**
2. **Ridge Classifier**
3. **Decision Tree**
4. **Random Forest**
5. **Gradient Boosting**
6. **Support Vector Machine (SVM)**
7. **Naive Bayes**

The best performing model is automatically selected and saved.

## 🔌 API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/` | API info |
| GET | `/health` | Health check |
| GET | `/model/info` | Model information |
| POST | `/predict` | Predict entities in text |
| POST | `/predict/batch` | Batch prediction |
| GET | `/entity-colors` | Color mapping for entities |

### Example API Call

```bash
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{"text": "Tesla TSLA reported revenue of $25 billion."}'
```

Response:
```json
{
  "text": "Tesla TSLA reported revenue of $25 billion.",
  "entities": [
    {"text": "Tesla", "label": "COMPANY", "start": 0, "end": 5},
    {"text": "TSLA", "label": "TICKER", "start": 6, "end": 10},
    {"text": "revenue", "label": "INDICATOR", "start": 20, "end": 27},
    {"text": "$", "label": "CURRENCY", "start": 31, "end": 32}
  ],
  "tokens": [...]
}
```

## 🎨 Frontend Features

- **Real-time entity extraction** from financial text
- **Color-coded highlighting** for different entity types
- **Interactive entity cards** showing extracted information
- **Token-level analysis** table
- **Sample text loader** for quick testing
- **API status indicator**

## 📦 Model Deployment

The trained model is saved as a ZIP file containing:
- `best_ner_model.pkl` - The trained model
- `tfidf_vectorizer.pkl` - Text vectorizer
- `label_mapping.json` - Label encodings
- `model_metadata.json` - Performance metrics

To deploy on another system:
1. Extract `financial_ner_model.zip` to the `models/` directory
2. Start the FastAPI backend
3. The model will be automatically loaded

## 🛠️ Technologies Used

- **Python 3.10+**
- **scikit-learn** - ML models
- **FastAPI** - Backend API
- **Uvicorn** - ASGI server
- **HTML/CSS/JavaScript** - Frontend

## 📈 Performance

Model performance is evaluated using:
- **Accuracy** - Overall correctness
- **F1 Macro** - Average F1 across all classes
- **F1 Weighted** - Weighted average F1

Results are visualized in the notebook with:
- Bar charts comparing metrics
- Heatmap of model performance

## 🔧 Customization

### Adding More Training Data

Edit the `sample_data` list in the notebook to add more examples:

```python
sample_data.append((
    "Your financial text here...",
    [(start, end, "ENTITY_TYPE"), ...]
))
```

### Adding New Entity Types

1. Add the new label to the `LABELS` list
2. Update `ENTITY_COLORS` in both backend and frontend
3. Retrain the model

## 📝 License

MIT License - Feel free to use for personal and commercial projects.

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.
