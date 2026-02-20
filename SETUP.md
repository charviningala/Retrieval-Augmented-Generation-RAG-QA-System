# Setup Instructions for RAG Q&A System

This document provides step-by-step instructions to set up and run the RAG Q&A System.

## System Requirements

- **Python**: 3.8 or higher
- **Node.js**: 14 or higher
- **npm**: 6 or higher (comes with Node.js)
- **RAM**: 8GB minimum (for loading ML models)
- **Disk Space**: 5GB+ (for model downloads)

## Installation Steps

### Step 1: Clone or Download Project

```bash
cd anlp_rag
```

Ensure your directory structure looks like this:
```
anlp_rag/
├── backend/
├── frontend/
├── cleaned_data/
├── test.ipynb
└── README.md
```

### Step 2: Backend Setup

#### 2.1 Navigate to Backend Directory
```bash
cd backend
```

#### 2.2 Create Virtual Environment (Recommended)

**Windows:**
```bash
python -m venv venv
venv\Scripts\activate
```

**macOS/Linux:**
```bash
python3 -m venv venv
source venv/bin/activate
```

#### 2.3 Install Dependencies
```bash
pip install -r requirements.txt
```

This will install:
- **flask** - Web framework
- **flask-cors** - Cross-origin resource sharing
- **sentence-transformers** - Embedding model
- **transformers** - LLM (FLAN-T5)
- **torch** - Deep learning framework
- **faiss-cpu** - Vector database
- **numpy** - Numerical computing

#### 2.4 Configure Environment (Optional)
Create a `.env` file:
```bash
cp .env .env.local  # or edit existing .env
```

#### 2.5 Test Backend
Run the Flask server:
```bash
python app.py
```

You should see:
```
Loading embedding model...
Loading LLM...
Loading and indexing documents...
✅ Models and index loaded successfully!
 * Running on http://127.0.0.1:5000
```

Go to `http://127.0.0.1:5000/api/health` in your browser. You should see:
```json
{
  "status": "ok",
  "model": "RAG Q&A System",
  "total_documents": 288
}
```

Keep this terminal running.

### Step 3: Frontend Setup

#### 3.1 Open New Terminal and Navigate to Frontend
```bash
cd frontend
```

#### 3.2 Install Dependencies
```bash
npm install
```

This will install:
- **react** - UI library
- **react-dom** - React rendering
- **axios** - HTTP client
- **react-markdown** - Markdown rendering
- **react-syntax-highlighter** - Code highlighting
- **react-scripts** - Build and development tools

#### 3.3 (Optional) Configure Environment
Create `.env.local`:
```bash
cp .env.example .env.local
```

Edit if needed (default values work):
```
REACT_APP_API_URL=http://localhost:5000
REACT_APP_DEBUG=true
```

#### 3.4 Start React Development Server
```bash
npm start
```

Your default browser should open automatically to `http://localhost:3000`.

## Running the Application

### Method 1: Manual Start (Recommended for Development)

**Terminal 1 - Backend:**
```bash
cd backend
python app.py
```

**Terminal 2 - Frontend:**
```bash
cd frontend
npm start
```

### Method 2: Automated Start

**Windows:**
```bash
start.bat
```

**macOS/Linux:**
```bash
bash start.sh
```

### Method 3: Docker (Optional)

If you have Docker installed, you can containerize the app. Create `Dockerfile` in root:

```dockerfile
FROM python:3.10-slim
WORKDIR /app
COPY backend/requirements.txt .
RUN pip install -r requirements.txt
COPY backend app.py
COPY cleaned_data ./cleaned_data
EXPOSE 5000
CMD ["python", "app.py"]
```

Then run:
```bash
docker build -t rag-qa .
docker run -p 5000:5000 rag-qa
```

## Verification

### Check Backend Health
```bash
curl http://localhost:5000/api/health
```

### Check Frontend Access
Open browser to `http://localhost:3000`

You should see:
- Header: "🤖 ML RAG Q&A System"
- Status indicator (green if connected)
- Query form on the left
- Welcome message on the right

## Test the System

1. Click on an example button ("Overfitting", "Linear Regression", etc.)
2. Or type your own question: "What is a neural network?"
3. Click "Ask"
4. You should see the RAG answer appear on the right
5. Click "Show Context" to see the retrieved context

## Troubleshooting

### Common Issues

#### Backend won't start
- **Error**: `ModuleNotFoundError: No module named 'flask'`
  - **Solution**: Install requirements: `pip install -r requirements.txt`
  
- **Error**: `Port 5000 already in use`
  - **Solution**: Kill process on port 5000 or change PORT in `.env`

#### Frontend won't connect
- **Error**: "Disconnected" status
  - **Solution**: Ensure Flask backend is running first
  
- **Error**: CORS errors in console
  - **Solution**: Verify `flask-cors` is installed

#### Models take too long to load
- This is normal first time (models download from HuggingFace)
- Subsequent runs will be faster
- Wait 2-5 minutes for first load

#### Out of memory errors
- **Solution**: Close other applications or upgrade RAM
- Minimum recommended: 8GB RAM
- Models require ~4-5GB when loaded

### Check Logs

**Backend logs:**
- Check terminal running `python app.py`
- Look for error messages

**Frontend logs:**
- Open browser DevTools (F12)
- Check Console tab for errors
- Check Network tab for API requests

### Port Configuration

If ports need changing:

**Backend (Flask):**
Edit `backend/.env`:
```
PORT=5001  # Change from 5000
```

**Frontend (React):**
Set environment variable before `npm start`:
```bash
# Windows
set PORT=3001

# macOS/Linux
export PORT=3001

npm start
```

Update proxy in `frontend/package.json`:
```json
"proxy": "http://localhost:5001"
```

## Performance Optimization

### For Development
Current setup is already optimized for development.

### For Production

1. **Build React**:
   ```bash
   cd frontend
   npm run build
   ```
   This creates optimized production build in `frontend/build/`

2. **Use Production WSGI Server**:
   ```bash
   pip install gunicorn
   cd backend
   gunicorn -w 4 -b 0.0.0.0:5000 app:app
   ```

3. **Serve React from Backend**:
   Copy `frontend/build/` contents to `backend/static/`

## API Documentation

### Health Check
```
GET /api/health
Response: { "status": "ok", "model": "RAG Q&A System", "total_documents": 288 }
```

### Query RAG System
```
POST /api/query
Payload: { "question": "What is Overfitting?" }
Response: { "question": "...", "answer": "...", "retrieved_context": "...", "success": true }
```

### Retrieve Documents
```
POST /api/retrieve
Payload: { "query": "machine learning", "k": 3 }
Response: { "query": "...", "results": [...], "count": 3, "success": true }
```

### List Concepts
```
GET /api/concepts
Response: { "concepts": [...], "total": 257, "success": true }
```

## Next Steps

1. ✅ Understand how RAG works (check notebook: `test.ipynb`)
2. ✅ Explore ML concepts available in the system
3. ✅ Customize the frontend UI as needed
4. ✅ Add more concepts to `cleaned_data/ML_COMPLETE_REFERENCE.txt`
5. ✅ Deploy to production

## Support

For issues:
1. Check terminal output for error messages
2. Review browser console (F12 → Console)
3. Check `README.md` for more details
4. Review logs from both backend and frontend

## Additional Resources

- **Flask**: https://flask.palletsprojects.com/
- **React**: https://react.dev/
- **FAISS**: https://github.com/facebookresearch/faiss
- **SentenceTransformers**: https://www.sbert.net/
- **HuggingFace**: https://huggingface.co/

---

**You're all set!** 🎉 Now you can ask questions about ML concepts using your RAG system.
