import React, { useState } from 'react';
import './ResponseDisplay.css';

function ResponseDisplay({ question, answer, baselineAnswer, context }) {
  const [showContext, setShowContext] = useState(false);

  const formatAnswer = (text) => {
    const lines = text.split('\n');
    return (
      <div className="answer-content">
        {lines.map((line, idx) => {
          if (line.startsWith('CONCEPT:')) {
            return (
              <h3 key={idx} className="concept-title">
                {line.replace('CONCEPT:', '').trim()}
              </h3>
            );
          } else if (line.startsWith('Definition:')) {
            return (
              <div key={idx} className="definition-section">
                <strong>Definition:</strong>{' '}
                {line.replace('Definition:', '').trim()}
              </div>
            );
          } else if (line.startsWith('Supporting Line:')) {
            return (
              <div key={idx} className="supporting-section">
                <strong>Supporting Line:</strong>{' '}
                {line.replace('Supporting Line:', '').trim()}
              </div>
            );
          } else if (line.startsWith('Formula:')) {
            return (
              <div key={idx} className="formula-section">
                <strong>Formula:</strong> <code>{line.replace('Formula:', '').trim()}</code>
              </div>
            );
          } else if (line.trim()) {
            return (
              <p key={idx} className="answer-text">
                {line}
              </p>
            );
          }
          return null;
        })}
      </div>
    );
  };

  const formatPlain = (text) => {
    const lines = text.split('\n').filter((line) => line.trim());
    return (
      <div className="baseline-content">
        {lines.map((line, idx) => (
          <p key={idx} className="answer-text">
            {line}
          </p>
        ))}
      </div>
    );
  };

  return (
    <div className="response-display">
      <div className="response-header">
        <h2>📝 Answer</h2>
        <button
          className="context-toggle"
          onClick={() => setShowContext(!showContext)}
        >
          {showContext ? '✕ Hide' : '👁 Show'} Context
        </button>
      </div>

      <div className="question-display">
        <strong>Your Question:</strong>
        <p>{question}</p>
      </div>

      <div className="answer-section">
        <strong>RAG Answer:</strong>
        {formatAnswer(answer)}
      </div>

      {baselineAnswer && (
        <div className="baseline-section">
          <strong>Non-RAG Baseline:</strong>
          {formatPlain(baselineAnswer)}
        </div>
      )}

      {showContext && (
        <div className="context-section">
          <strong>Retrieved Context (First 500 chars):</strong>
          <div className="context-text">{context}</div>
        </div>
      )}
    </div>
  );
}

export default ResponseDisplay;
