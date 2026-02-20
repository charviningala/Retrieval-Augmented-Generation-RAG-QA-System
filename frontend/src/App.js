import React, { useState, useEffect } from 'react';
import axios from 'axios';
import './App.css';
import QueryForm from './components/QueryForm';
import ResponseDisplay from './components/ResponseDisplay';
import MetricsPanel from './components/MetricsPanel';
import TopicsList from './components/TopicsList';

function App() {
  const [question, setQuestion] = useState('');
  const [answer, setAnswer] = useState('');
  const [context, setContext] = useState('');
  const [baselineAnswer, setBaselineAnswer] = useState('');
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');
  const [metrics, setMetrics] = useState(null);
  const [history, setHistory] = useState([]);
  const [systemStatus, setSystemStatus] = useState('checking');

  // Check API health on mount
  useEffect(() => {
    checkHealth();
  }, []);

  const checkHealth = async () => {
    try {
      const response = await axios.get('/api/health');
      setSystemStatus('connected');
    } catch (err) {
      setSystemStatus('disconnected');
      setError('Backend API is not running. Start the Flask server first.');
    }
  };

  const handleQuery = async (q) => {
    if (!q.trim()) {
      setError('Please enter a question');
      return;
    }

    setLoading(true);
    setError('');
    setAnswer('');
    setContext('');
    setBaselineAnswer('');

    try {
      const startTime = performance.now();
      const ragPromise = axios.post('/api/query', { question: q });
      const baselinePromise = axios.post('/api/baseline', { question: q });
      const [ragResponse, baselineResponse] = await Promise.all([
        ragPromise,
        baselinePromise,
      ]);
      const endTime = performance.now();

      const baselineData = baselineResponse?.data;

      if (ragResponse.data.success) {
        setQuestion(q);
        setAnswer(ragResponse.data.answer);
        setContext(ragResponse.data.retrieved_context);

        if (baselineData?.success) {
          setBaselineAnswer(baselineData.answer);
        }

        // Calculate metrics
        setMetrics({
          questionLength: q.length,
          rag: {
            latency_ms: ragResponse.data.metrics?.latency_ms ?? Number((endTime - startTime).toFixed(2)),
            answerLength: ragResponse.data.metrics?.answer_length ?? ragResponse.data.answer.length,
            contextLength: ragResponse.data.metrics?.context_length ?? 0,
            model: ragResponse.data.metrics?.model ?? 'RAG',
            rouge1: ragResponse.data.metrics?.rouge1 ?? 0,
            rougeL: ragResponse.data.metrics?.rougeL ?? 0,
            semantic_similarity: ragResponse.data.metrics?.semantic_similarity ?? 0,
          },
          baseline: baselineData?.success
            ? {
                latency_ms: baselineData.metrics?.latency_ms ?? null,
                answerLength: baselineData.metrics?.answer_length ?? baselineData.answer.length,
                contextLength: baselineData.metrics?.context_length ?? 0,
                model: baselineData.metrics?.model ?? 'Baseline',
                rouge1: baselineData.metrics?.rouge1 ?? 0,
                rougeL: baselineData.metrics?.rougeL ?? 0,
                semantic_similarity: baselineData.metrics?.semantic_similarity ?? 0,
              }
            : null,
        });

        // Add to history
        setHistory([
          {
            question: q,
            answer: ragResponse.data.answer,
            timestamp: new Date(),
          },
          ...history.slice(0, 9),
        ]);
      } else {
        setError('Failed to get response from server');
      }
    } catch (err) {
      setError(err.response?.data?.error || 'Error querying the RAG system');
      console.error('Query error:', err);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="app">
      <header className="app-header">
        <div className="header-content">
          <h1>🤖 ML RAG Q&A System</h1>
          <p>Ask questions about Machine Learning concepts</p>
          <div className={`status ${systemStatus}`}>
            {systemStatus === 'connected' ? '✓ Connected' : '✗ Disconnected'}
          </div>
        </div>
      </header>

      <main className="app-main">
        <div className="container">
          <div className="left-panel">
            <QueryForm onQuery={handleQuery} loading={loading} />
            {error && <div className="error-message">{error}</div>}
          </div>

          <div className="right-panel">
            {loading ? (
              <div className="loading">
                <div className="spinner"></div>
                <p>Processing your question...</p>
              </div>
            ) : answer ? (
              <>
                <ResponseDisplay
                  question={question}
                  answer={answer}
                  baselineAnswer={baselineAnswer}
                  context={context}
                />
                {metrics && <MetricsPanel metrics={metrics} />}
              </>
            ) : (
              <>
                <div className="welcome-message">
                  <h2>Welcome to ML RAG Q&A</h2>
                  <p>Enter a question about machine learning concepts to get started.</p>
                  <p className="example">
                    Example: "What is Overfitting?" or "Define Linear Regression"
                  </p>
                </div>
                <TopicsList onTopicClick={(topic) => {
                  const query = `What is ${topic}?`;
                  setQuestion(query);
                  handleQuery(query);
                }} />
              </>
            )}
          </div>
        </div>

        {history.length > 0 && (
          <div className="history-section">
            <h3>History</h3>
            <div className="history-list">
              {history.map((item, idx) => (
                <div
                  key={idx}
                  className="history-item"
                  onClick={() => handleQuery(item.question)}
                >
                  <p className="history-question">{item.question}</p>
                  <p className="history-time">
                    {item.timestamp.toLocaleTimeString()}
                  </p>
                </div>
              ))}
            </div>
          </div>
        )}
      </main>
    </div>
  );
}

export default App;
