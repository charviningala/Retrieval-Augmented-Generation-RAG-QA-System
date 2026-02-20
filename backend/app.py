from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import re
import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import faiss
import os
import time
from sklearn.metrics.pairwise import cosine_similarity
from rouge_score import rouge_scorer

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
FRONTEND_BUILD_DIR = os.path.join(BASE_DIR, '..', 'frontend', 'build')

# Serve frontend from build folder
app = Flask(__name__, static_folder=FRONTEND_BUILD_DIR, static_url_path='')
CORS(app)

# ===============================
# LOAD MODELS & EMBEDDINGS
# ===============================

print("Loading embedding model...")
embed_model = SentenceTransformer("all-MiniLM-L6-v2")

print("Loading LLM...")
tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-large")
model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-large")
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

print("Loading and indexing documents...")
# Load documents from cleaned_data
documents = []

with open(os.path.join(BASE_DIR, "..", "cleaned_data", "ML_COMPLETE_REFERENCE.txt"), "r", encoding="utf-8") as f:
    detailed_content = f.read()

concept_pattern = r'CONCEPT: [^\n]+(?:\nDefinition:[^\n]+)?(?:\nSupporting Line:[^\n]+)?(?:\nFormula:[^\n]+)?'
concept_blocks = re.findall(concept_pattern, detailed_content)

detailed_sections = re.split(r'={80,}', detailed_content)
detailed_sections = [s.strip() for s in detailed_sections if s.strip() and len(s.strip()) > 50]

with open(os.path.join(BASE_DIR, "..", "cleaned_data", "ML_NOTES_QUICK_REFERENCE.txt"), "r", encoding="utf-8") as f:
    quick_ref_content = f.read()

quick_ref_sections = re.split(r'={80,}', quick_ref_content)
quick_ref_sections = [s.strip() for s in quick_ref_sections if s.strip() and len(s.strip()) > 50]

documents.extend(concept_blocks)
documents.extend(detailed_sections)
documents.extend(quick_ref_sections)

print(f"Total documents loaded: {len(documents)}")

# Build FAISS index
embeddings = embed_model.encode(documents, show_progress_bar=True)
embeddings = np.array(embeddings).astype("float32")
dimension = embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(embeddings)

print("✅ Models and index loaded successfully!")

# ===============================
# RAG FUNCTIONS
# ===============================

def retrieve(query, k=1):
    """Retrieve documents from FAISS index"""
    query_embedding = embed_model.encode([query]).astype("float32")
    distances, indices = index.search(query_embedding, k)
    return [documents[i] for i in indices[0]]

def extract_definition_reference(context):
    """Extract just the Definition line from CONCEPT block as reference for scoring"""
    lines = context.split('\n')
    for i, line in enumerate(lines):
        if line.strip().startswith("Definition:"):
            # Return just the definition line
            return line.replace("Definition:", "").strip()
    # Fallback: return first non-empty line
    for line in lines:
        if line.strip() and not line.strip().startswith("CONCEPT:"):
            return line.strip()
    return context[:100]  # Last resort

def generate_rag(query, k=1):
    """RAG generation with concept extraction"""
    retrieved_docs = retrieve(query, k=k)
    context = "\n\n".join(retrieved_docs)
    
    # Extract concept name from query
    concept_name = query.replace("What is ", "").replace("Define ", "").replace("?", "").strip()
    
    lines = context.split('\n')
    answer_lines = []
    found_concept = False
    
    # Try exact match
    for i, line in enumerate(lines):
        if f"CONCEPT: {concept_name}" in line:
            found_concept = True
            answer_lines = lines[i:min(i+4, len(lines))]
            break
    
    # Try partial match
    if not found_concept and concept_name:
        first_word = concept_name.split()[0].lower()
        for i, line in enumerate(lines):
            if "CONCEPT:" in line and first_word in line.lower():
                found_concept = True
                answer_lines = lines[i:min(i+4, len(lines))]
                break
    
    if found_concept and answer_lines:
        return "\n".join(answer_lines)
    
    # Find first CONCEPT block
    for i, line in enumerate(lines):
        if "CONCEPT:" in line:
            answer_lines = lines[i:min(i+4, len(lines))]
            return "\n".join(answer_lines)
    
    # Fallback to model generation
    prompt = f"""From this context, answer: {query}

CONTEXT:
{context}

ANSWER:"""
    
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024).to(device)
    outputs = model.generate(
        **inputs,
        max_new_tokens=200,
        temperature=0.2,
        do_sample=False,
        repetition_penalty=1.2
    )
    
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

def generate_baseline(query):
    """Non-RAG baseline generation (direct model response)"""
    prompt = f"""Answer the question clearly and concisely:

QUESTION:
{query}

ANSWER:"""

    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512).to(device)
    outputs = model.generate(
        **inputs,
        max_new_tokens=200,
        temperature=0.3,
        do_sample=False,
        repetition_penalty=1.1
    )

    return tokenizer.decode(outputs[0], skip_special_tokens=True)

# Ground truth references - independent of retrieved documents to avoid bias
GROUND_TRUTH_REFERENCES = {
    "overfitting": "Overfitting is when a model has low training error but high test error because it memorizes training data instead of learning general patterns.",
    "linear regression": "Linear regression is a supervised learning algorithm used to predict continuous values by fitting a linear relationship between input features and target variable.",
    "logistic regression": "Logistic regression is a supervised learning algorithm used for binary classification that predicts probabilities using a sigmoid function.",
    "gradient descent": "Gradient descent is an optimization algorithm that iteratively updates model parameters by moving in the direction of steepest descent of the loss function.",
    "neural network": "A neural network is a machine learning model composed of interconnected layers of neurons that can learn complex patterns through backpropagation.",
    "deep learning": "Deep learning uses neural networks with multiple hidden layers to automatically learn hierarchical representations of data.",
    "backpropagation": "Backpropagation is an algorithm for training neural networks by computing gradients of the loss function with respect to weights using the chain rule.",
    "regularization": "Regularization is a technique to prevent overfitting by adding a penalty term to the loss function that discourages complex models.",
    "cross-validation": "Cross-validation is a technique to evaluate model performance by splitting data into multiple folds and training on different subsets.",
    "feature engineering": "Feature engineering is the process of creating, selecting, and transforming input variables to improve model performance.",
    "clustering": "Clustering is an unsupervised learning technique that groups similar data points together based on their features.",
    "classification": "Classification is a supervised learning task that assigns discrete labels to input data based on learned patterns.",
    "regression": "Regression is a supervised learning task that predicts continuous numerical values based on input features.",
    "ensemble methods": "Ensemble methods combine multiple models to improve predictive performance by reducing variance, bias, or improving predictions.",
    "decision tree": "A decision tree is a supervised learning model that makes predictions by learning decision rules from data features.",
    "random forest": "Random forest is an ensemble method that combines multiple decision trees trained on random subsets of data and features.",
    "support vector machine": "Support vector machine is a supervised learning algorithm that finds the optimal hyperplane to separate different classes.",
    "dimensionality reduction": "Dimensionality reduction reduces the number of input features while preserving important information to improve efficiency and reduce overfitting.",
    "activation function": "An activation function introduces non-linearity into neural networks, allowing them to learn complex patterns.",
    "loss function": "A loss function measures the difference between predicted and actual values, guiding the model optimization process.",
}

def get_ground_truth_reference(question):
    """Extract ground truth reference for a given question"""
    # Normalize question to extract concept
    question_lower = question.lower()
    
    # Try to match against known concepts
    for concept, reference in GROUND_TRUTH_REFERENCES.items():
        if concept in question_lower:
            return reference
    
    # Default: return a generic reference
    return "A machine learning concept that helps build predictive models."

def calculate_rouge_scores(reference, hypothesis):
    """Calculate ROUGE-1 and ROUGE-L scores using official rouge-score library"""
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rougeL'], use_stemmer=True)
    scores = scorer.score(reference, hypothesis)
    
    return {
        "rouge1": round(scores['rouge1'].fmeasure, 3),
        "rougeL": round(scores['rougeL'].fmeasure, 3)
    }

def calculate_semantic_similarity(reference, answer):
    """Calculate semantic similarity between reference and generated answer"""
    try:
        ref_emb = embed_model.encode([reference])
        answer_emb = embed_model.encode([answer])
        similarity = cosine_similarity(ref_emb, answer_emb)[0][0]
        return round(float(similarity), 3)
    except:
        return 0.0

# ===============================
# ROUTES
# ===============================

@app.route('/api/health', methods=['GET'])
def health():
    """Health check endpoint"""
    return jsonify({
        "status": "ok",
        "model": "RAG Q&A System",
        "total_documents": len(documents)
    })

@app.route('/api/query', methods=['POST'])
def query_endpoint():
    """Main RAG query endpoint"""
    try:
        data = request.json
        question = data.get('question', '').strip()
        
        if not question:
            return jsonify({"error": "Question cannot be empty"}), 400
        
        start_time = time.perf_counter()
        # Retrieve documents
        retrieved_docs = retrieve(question, k=1)
        
        # Generate RAG answer
        answer = generate_rag(question)
        latency_ms = (time.perf_counter() - start_time) * 1000
        
        # Calculate evaluation metrics using ground truth reference (unbiased)
        reference = get_ground_truth_reference(question)
        rouge_scores = calculate_rouge_scores(reference, answer)
        semantic_sim = calculate_semantic_similarity(reference, answer)
        
        return jsonify({
            "question": question,
            "answer": answer,
            "retrieved_context": retrieved_docs[0][:500] if retrieved_docs else "No context found",
            "metrics": {
                "latency_ms": round(latency_ms, 2),
                "answer_length": len(answer),
                "context_length": len(retrieved_docs[0]) if retrieved_docs else 0,
                "model": "RAG (FAISS + FLAN-T5)",
                "rouge1": rouge_scores["rouge1"],
                "rougeL": rouge_scores["rougeL"],
                "semantic_similarity": semantic_sim
            },
            "success": True
        }), 200
        
    except Exception as e:
        return jsonify({
            "error": str(e),
            "success": False
        }), 500

@app.route('/api/baseline', methods=['POST'])
def baseline_endpoint():
    """Non-RAG baseline endpoint"""
    try:
        data = request.json
        question = data.get('question', '').strip()

        if not question:
            return jsonify({"error": "Question cannot be empty"}), 400

        start_time = time.perf_counter()
        # Generate baseline answer (no retrieval used)
        answer = generate_baseline(question)
        latency_ms = (time.perf_counter() - start_time) * 1000

        # Calculate evaluation metrics using same ground truth reference as RAG
        reference = get_ground_truth_reference(question)
        rouge_scores = calculate_rouge_scores(reference, answer)
        semantic_sim = calculate_semantic_similarity(reference, answer)

        return jsonify({
            "question": question,
            "answer": answer,
            "metrics": {
                "latency_ms": round(latency_ms, 2),
                "answer_length": len(answer),
                "context_length": 0,
                "model": "Baseline (FLAN-T5 only)",
                "rouge1": rouge_scores["rouge1"],
                "rougeL": rouge_scores["rougeL"],
                "semantic_similarity": semantic_sim
            },
            "success": True
        }), 200

    except Exception as e:
        return jsonify({
            "error": str(e),
            "success": False
        }), 500

@app.route('/api/retrieve', methods=['POST'])
def retrieve_endpoint():
    """Retrieve similar documents"""
    try:
        data = request.json
        query = data.get('query', '').strip()
        k = data.get('k', 3)
        
        if not query:
            return jsonify({"error": "Query cannot be empty"}), 400
        
        retrieved = retrieve(query, k=k)
        
        return jsonify({
            "query": query,
            "results": retrieved,
            "count": len(retrieved),
            "success": True
        }), 200
        
    except Exception as e:
        return jsonify({
            "error": str(e),
            "success": False
        }), 500

@app.route('/api/concepts', methods=['GET'])
def list_concepts():
    """List all available concepts"""
    try:
        concepts = []
        for doc in documents[:257]:  # First 257 are concept blocks
            if doc.startswith("CONCEPT:"):
                concept_name = doc.split('\n')[0].replace("CONCEPT:", "").strip()
                concepts.append(concept_name)
        
        return jsonify({
            "concepts": concepts[:50],  # Return first 50
            "total": len(concepts),
            "success": True
        }), 200
        
    except Exception as e:
        return jsonify({
            "error": str(e),
            "success": False
        }), 500

# Serve static files from frontend/build
@app.route('/')
def serve_index():
    """Serve index.html"""
    return send_from_directory(app.static_folder, 'index.html')

@app.route('/<path:path>')
def serve_static(path):
    """Serve static files or fallback to index.html for frontend routing"""
    # Check if file exists in static folder
    if os.path.exists(os.path.join(app.static_folder, path)):
        return send_from_directory(app.static_folder, path)
    # For any non-existent routes, serve index.html (for React Router)
    return send_from_directory(app.static_folder, 'index.html')

if __name__ == '__main__':
    app.run(debug=True, host='127.0.0.1', port=5000)
