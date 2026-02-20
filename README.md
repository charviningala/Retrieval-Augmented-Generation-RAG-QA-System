# RAG Q&A System - ML Concepts

A full-stack Retrieval-Augmented Generation (RAG) Q&A application for answering Machine Learning concept questions. The Flask backend serves both the API and the built React frontend from a single port.

## 🎯 Project Structure

```
anlp_rag/
├── backend/                    # Flask API server
│   ├── app.py                 # Main Flask application
│   ├── requirements.txt        # Python dependencies
│   └── .env                     # Environment configuration
├── frontend/                   # React web application
│   ├── public/
│   │   └── index.html
│   ├── build/                   # Production build served by Flask
│   ├── src/
│   │   ├── App.js
│   │   ├── App.css
│   │   ├── index.js
│   │   └── components/
│   │       ├── QueryForm.js
│   │       ├── ResponseDisplay.js
│   │       └── MetricsPanel.js
│   └── package.json
├── test.ipynb                 # Original RAG system notebook
└── cleaned_data/              # ML reference documents
    ├── ML_COMPLETE_REFERENCE.txt
    └── ML_NOTES_QUICK_REFERENCE.txt
```

## 🚀 Quick Start

### Prerequisites
- Python 3.8+
- Node.js 18+
- npm (comes with Node.js)

### Backend Setup

1. **Navigate to backend directory**
   ```bash
   cd backend
   ```

2. **Create virtual environment (optional but recommended)**
   ```bash
   python -m venv venv
   # Windows
   venv\Scripts\activate
   # macOS/Linux
   source venv/bin/activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **(Optional) Start Flask server**
   You'll start it after the frontend build step below.

### Frontend Build (served by Flask)

1. **Navigate to frontend directory** (in a new terminal)
   ```bash
   cd frontend
   ```

2. **Install dependencies**
   ```bash
   npm install
   ```

3. **Build the React frontend**
   ```bash
   npm run build
   ```

### Run Integrated App

From the project root:
```bash
python backend/app.py
```

Open `http://localhost:5000` in your browser.

## 🔗 API Endpoints

### Health Check
```
GET /api/health
```
Returns system status and document count.

### Query RAG System
```
POST /api/query
Content-Type: application/json

{
  "question": "What is Overfitting?"
}
```

Response:
```json
{
  "question": "What is Overfitting?",
  "answer": "CONCEPT: Overfitting\nDefinition: ...",
  "retrieved_context": "...",
  "success": true
}
```

### Retrieve Similar Documents
```
POST /api/retrieve
Content-Type: application/json

{
  "query": "machine learning concepts",
  "k": 3
}
```

### List Available Concepts
```
GET /api/concepts
```

## 🧠 How It Works

1. **Document Loading**: ML concepts are loaded from `ML_COMPLETE_REFERENCE.txt` and split into 257 fine-grained concept blocks
2. **Embedding**: SentenceTransformer converts all concept blocks into 384-dimensional vectors
3. **Indexing**: FAISS creates a fast nearest-neighbor index for retrieval
4. **Query Processing**: User question is embedded and matched against concept blocks
5. **Answer Extraction**: Matched concept block is extracted with Definition, Supporting Line, and Formula
6. **Fallback**: If exact concept not found, FLAN-T5-Large model generates answer using context

## 📊 Performance Metrics

Current system performance on 4 test QA pairs:

| Metric | Non-RAG | RAG | Improvement |
|--------|---------|-----|------------|
| ROUGE-1 | 0.2981 | 0.2498 | -0.048 |
| ROUGE-L | 0.2526 | 0.2294 | -0.023 |
| **Semantic Similarity** | 0.6955 | **0.7728** | **+0.077** ✓ |

RAG system excels in semantic understanding with +0.077 improvement in semantic similarity.

## 🎨 Features

- ✅ Real-time RAG-based QA system
- ✅ Fine-grained concept extraction (257 concept blocks)
- ✅ Semantic document retrieval with FAISS
- ✅ Beautiful, responsive React UI
- ✅ Query history tracking
- ✅ Response metrics (latency, length, etc.)
- ✅ Context visibility toggle
- ✅ Example questions for easy testing
- ✅ System status indicator

## 🔧 Configuration

### Backend Environment Variables (.env)
```
FLASK_ENV=development
FLASK_APP=app.py
FLASK_DEBUG=True
HOST=127.0.0.1
PORT=5000
```

### Frontend API Configuration
The frontend uses relative `/api` paths. When served by Flask, no proxy is required.

## 📝 Example Questions

- "What is Overfitting?"
- "Define Linear Regression"
- "What is Logistic Regression?"
- "What is Silhouette Score?"
- "Explain Bias-Variance Tradeoff"
- "What is K-Fold Cross-Validation?"

## 🐛 Troubleshooting

### Backend Issues
- **Models not loading**: Ensure transformers library is installed: `pip install transformers torch`
- **FAISS error**: `pip install faiss-cpu`
- **Port in use**: Change PORT in .env file

### Frontend Issues
- **API not connecting**: Check Flask server is running on port 5000
- **CORS error**: Ensure `flask-cors` is installed on backend
- **Blank page**: Run `npm run build` so `frontend/build` exists
- **Blank responses**: Check browser console and Flask logs for errors

## 📚 ML Concepts Covered

The system has comprehensive definitions for 130+ ML concepts including:
- Overfitting, Underfitting
- Linear/Logistic Regression
- Bias-Variance Tradeoff
- Cross-Validation techniques
- Clustering metrics
- Regularization methods
- And many more...

## 🚀 Deployment

### Docker (Optional)
Create `Dockerfile` in root directory for containerization.

### Production Tips
- Use a production WSGI server (gunicorn, waitress)
- Set Flask debug to False
- Use environment variables for sensitive config
- Enable proper CORS settings
- Run React build: `npm run build`

## 📚 References

- [Flask Documentation](https://flask.palletsprojects.com/)
- [React Documentation](https://react.dev/)
- [SentenceTransformers](https://www.sbert.net/)
- [FAISS](https://github.com/facebookresearch/faiss)
- [Hugging Face Transformers](https://huggingface.co/docs/transformers/)

## 📄 License

This project uses open-source models and libraries. See individual licenses for details.
