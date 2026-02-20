import React from 'react';
import './TopicsList.css';

function TopicsList({ onTopicClick }) {
  const topics = [
    'Overfitting',
    'Linear Regression',
    'Logistic Regression',
    'Gradient Descent',
    'Backpropagation',
    'Regularization',
    'Feature Engineering',
    'Supervised Learning',
    'Unsupervised Learning',
    'Reinforcement Learning',
    'Cross-Entropy Loss',
    'Sigmoid Activation',
    'ReLU Activation',
    'Principal Component Analysis (PCA)',
    'Generalization',
    'Maximum Likelihood',
    'Bayes Rule',
    'Covariance Matrix Σ',
    'Decision Boundary',
    'Learning Rate (Step Size) η',
    'Basis Functions',
    'Normal Equations',
    'Binary Classification',
    'Multi-class Classification',
    'Conditional Probability P(A|B)',
    'Gaussian (Normal) Distribution',
    'Objective Function',
    'Universal Approximation',
    'Convexity',
    'Weight Decay (L2 Regularization)',
  ];

  return (
    <div className="topics-container">
      <div className="topics-header">
        <h3>Explore Topics</h3>
        <p>Click any topic to ask a question about it</p>
      </div>
      <div className="topics-grid">
        {topics.map((topic, index) => (
          <button
            key={index}
            className="topic-tag"
            onClick={() => onTopicClick(topic)}
            title={`Ask about ${topic}`}
          >
            {topic}
          </button>
        ))}
      </div>
    </div>
  );
}

export default TopicsList;
