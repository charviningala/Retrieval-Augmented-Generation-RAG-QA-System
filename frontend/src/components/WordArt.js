import React, { useEffect, useState } from 'react';
import './WordArt.css';

function WordArt() {
  const [words, setWords] = useState([]);

  const mlTopics = [
    'Neural Network',
    'Deep Learning',
    'Overfitting',
    'Regularization',
    'Gradient Descent',
    'Backpropagation',
    'Cross-Validation',
    'Clustering',
    'Classification',
    'Regression',
    'Dimensionality Reduction',
    'Feature Engineering',
    'Ensemble Methods',
    'Optimization',
    'Loss Function',
    'Activation Function',
    'Hyperparameter Tuning',
    'Transfer Learning',
    'Semi-Supervised Learning',
    'Reinforcement Learning',
    'Attention Mechanism',
    'Convolution',
    'Recurrence',
    'Embedding',
    'Tokenization',
    'Normalization',
    'Batch Processing',
    'Dropout',
    'Momentum',
    'Adam Optimizer',
    'Precision & Recall',
    'F1 Score',
    'ROC Curve',
    'Confusion Matrix',
    'Bias-Variance',
    'Data Augmentation',
    'Fine-tuning',
    'Zero-shot Learning',
    'Few-shot Learning',
    'Meta-learning',
  ];

  useEffect(() => {
    // Use a more aggressive spiral with quadrant-based distribution
    const generatedWords = mlTopics.map((topic, index) => {
      // Using Ulam spiral pattern for cleaner distribution
      const n = index + 1;
      const layer = Math.floor((Math.sqrt(n) - 1) / 2) + 1;
      const side = layer * 2;
      const sidePos = n - (layer * 2 - 1) ** 2;
      
      let x, y;
      if (sidePos <= side) {
        // Right side
        x = 50 + layer * 15;
        y = 50 + (sidePos - layer) * 15;
      } else if (sidePos <= 2 * side) {
        // Top side
        x = 50 - (sidePos - side - layer) * 15;
        y = 50 + layer * 15;
      } else if (sidePos <= 3 * side) {
        // Left side
        x = 50 - layer * 15;
        y = 50 - (sidePos - 2 * side - layer) * 15;
      } else {
        // Bottom side
        x = 50 + (sidePos - 3 * side - layer) * 15;
        y = 50 - layer * 15;
      }
      
      // Clamp to container with good margins
      const left = Math.max(4, Math.min(x, 92));
      const top = Math.max(4, Math.min(y, 91));
      
      const size = 0.75 + (index % 3) * 0.08; // Very consistent sizing: 0.75, 0.83, 0.91rem
      const delay = (index * 20) / 1000;
      const duration = 7 + (index % 4) * 0.5;
      
      return {
        id: index,
        text: topic,
        size,
        delay,
        duration,
        left,
        top,
      };
    });
    setWords(generatedWords);
  }, []);

  return (
    <div className="word-art-container">
      <div className="word-art-background"></div>
      <div className="word-art">
        {words.map((word) => (
          <span
            key={word.id}
            className="word-art-item"
            style={{
              fontSize: `${word.size}rem`,
              left: `${word.left}%`,
              top: `${word.top}%`,
              '--delay': `${word.delay}s`,
              '--duration': `${word.duration}s`,
            }}
          >
            {word.text}
          </span>
        ))}
      </div>
    </div>
  );
}

export default WordArt;
