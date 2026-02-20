import React, { useState } from 'react';
import './QueryForm.css';

function QueryForm({ onQuery, loading }) {
  const [input, setInput] = useState('');

  const handleSubmit = (e) => {
    e.preventDefault();
    if (input.trim()) {
      onQuery(input);
      setInput('');
    }
  };

  const handleExampleClick = (example) => {
    setInput(example);
  };

  return (
    <form onSubmit={handleSubmit} className="query-form">
      <h2>Ask a Question</h2>
      
      <div className="input-group">
        <textarea
          value={input}
          onChange={(e) => setInput(e.target.value)}
          placeholder="e.g., What is Overfitting? or Define Linear Regression"
          disabled={loading}
          rows="4"
          className="query-input"
        />
        <button
          type="submit"
          disabled={loading || !input.trim()}
          className="query-button"
        >
          {loading ? 'Searching...' : 'Ask'}
        </button>
      </div>

      <div className="examples">
        <h4>Try these examples:</h4>
        <div className="example-buttons">
          <button
            type="button"
            onClick={() => handleExampleClick('What is Overfitting?')}
            className="example-btn"
          >
            Overfitting
          </button>
          <button
            type="button"
            onClick={() => handleExampleClick('Define Linear Regression')}
            className="example-btn"
          >
            Linear Regression
          </button>
          <button
            type="button"
            onClick={() => handleExampleClick('What is Logistic Regression?')}
            className="example-btn"
          >
            Logistic Regression
          </button>
          <button
            type="button"
            onClick={() => handleExampleClick('What is Silhouette Score?')}
            className="example-btn"
          >
            Silhouette Score
          </button>
        </div>
      </div>
    </form>
  );
}

export default QueryForm;
