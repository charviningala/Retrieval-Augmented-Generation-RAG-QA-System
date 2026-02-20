import React from 'react';
import './MetricsPanel.css';

function MetricsPanel({ metrics }) {
  const rag = metrics.rag;
  const baseline = metrics.baseline;

  const getScoreBadge = (ragValue, baselineValue) => {
    if (!baselineValue) return null;
    const diff = ragValue - baselineValue;
    if (diff > 0.02) {
      return <span className="score-badge positive">↑ Better</span>;
    } else if (diff < -0.02) {
      return <span className="score-badge negative">↓ Lower</span>;
    }
    return null;
  };

  return (
    <div className="metrics-panel">
      <h3>📊 Evaluation Metrics</h3>

      <div className="metrics-comparison">
        <div className="pipeline-metrics">
          <div className="pipeline-header">
            <h4>🎯 RAG Pipeline</h4>
            <span className="pipeline-model">{rag.model}</span>
          </div>
          <div className="metrics-grid">
            <div className="metric-card">
              <div className="metric-label">ROUGE-1</div>
              <div className="metric-value">{rag.rouge1}</div>
              {getScoreBadge(rag.rouge1, baseline?.rouge1)}
            </div>
            <div className="metric-card">
              <div className="metric-label">ROUGE-L</div>
              <div className="metric-value">{rag.rougeL}</div>
              {getScoreBadge(rag.rougeL, baseline?.rougeL)}
            </div>
            <div className="metric-card">
              <div className="metric-label">Semantic Similarity</div>
              <div className="metric-value">{rag.semantic_similarity}</div>
              {getScoreBadge(rag.semantic_similarity, baseline?.semantic_similarity)}
            </div>
            <div className="metric-card">
              <div className="metric-label">Latency</div>
              <div className="metric-value">{rag.latency_ms}ms</div>
            </div>
          </div>
        </div>

        {baseline && (
          <div className="pipeline-metrics">
            <div className="pipeline-header">
              <h4>⚙️ Baseline</h4>
              <span className="pipeline-model">{baseline.model}</span>
            </div>
            <div className="metrics-grid">
              <div className="metric-card">
                <div className="metric-label">ROUGE-1</div>
                <div className="metric-value">{baseline.rouge1}</div>
              </div>
              <div className="metric-card">
                <div className="metric-label">ROUGE-L</div>
                <div className="metric-value">{baseline.rougeL}</div>
              </div>
              <div className="metric-card">
                <div className="metric-label">Semantic Similarity</div>
                <div className="metric-value">{baseline.semantic_similarity}</div>
              </div>
              <div className="metric-card">
                <div className="metric-label">Latency</div>
                <div className="metric-value">{baseline.latency_ms}ms</div>
              </div>
            </div>
          </div>
        )}
      </div>

      {baseline && (
        <div className="metrics-summary">
          <h4>Summary</h4>
          <div className="summary-grid">
            <div className="summary-item">
              <span>ROUGE-1 Advantage</span>
              <strong>{(rag.rouge1 - baseline.rouge1).toFixed(3)}</strong>
            </div>
            <div className="summary-item">
              <span>Semantic Similarity Advantage</span>
              <strong>{(rag.semantic_similarity - baseline.semantic_similarity).toFixed(3)}</strong>
            </div>
            <div className="summary-item">
              <span>Speed</span>
              <strong>
                {((baseline.latency_ms - rag.latency_ms) / baseline.latency_ms * 100).toFixed(0)}%
                {((baseline.latency_ms - rag.latency_ms) / baseline.latency_ms * 100) > 0 ? ' faster' : ' slower'}
              </strong>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}

export default MetricsPanel;
