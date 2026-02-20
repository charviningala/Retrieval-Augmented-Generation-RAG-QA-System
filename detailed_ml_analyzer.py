import json
import re

class DetailedMLAnalyzer:
    """Create detailed analysis with true equations and better summaries"""
    
    def __init__(self, json_file):
        with open(json_file) as f:
            self.data = json.load(f)
    
    def extract_real_equations(self):
        """Extract genuine mathematical equations using advanced patterns"""
        equations = {}
        
        # Mathematical equation patterns
        patterns = [
            # Pattern 1: f(x) = ... 
            (r'([a-z]\([^)]+\)\s*=\s*[^\n.;]+)', 'Function definition'),
            # Pattern 2: Variable = Expression with math operators
            (r'([a-z_]\s*=\s*[a-zA-Z0-9_\s\(\)\[\]\+\-\*\/\.]+(?:[a-z_]\([^)]*\))?)', 'Assignment'),
            # Pattern 3: Symbols with subscripts/superscripts (x_i, w^2, etc.)
            (r'([a-z][\w_\^\']+\s*[=≈∝∈]\s*[^.]*)', 'Variable relation'),
            # Pattern 4: Mathematical notation (∑, ∏, ∫, etc.)
            (r'.*[∑∏∫∂∇].*', 'Advanced math notation'),
            # Pattern 5: Matrix/vector notation (||...||, [...])
            (r'(\|\|[^|]+\|\||\[[^\]]+\])', 'Vector/Matrix'),
        ]
        
        eqs_found = []
        for entry in self.data:
            content = entry.get('content', '')
            page = entry.get('page_number', 'N/A')
            
            for pattern, eq_type in patterns:
                matches = re.finditer(pattern, content, re.IGNORECASE)
                for match in matches:
                    eq = match.group(1).strip()
                    if len(eq) > 5 and not any(common in eq.lower() for common in ['and', 'the', 'that', 'this']):
                        eqs_found.append({
                            'equation': eq[:100],
                            'type': eq_type,
                            'page': page,
                            'full_context': content[max(0, match.start()-30):min(len(content), match.end()+30)]
                        })
        
        return eqs_found
    
    def analyze_topics(self):
        """Analyze topics with key concepts and equations"""
        topics = {
            'Linear Regression': {
                'keywords': ['Linear Regression', 'least-squares', 'weights', 'bias'],
                'key_equations': ['y = wx + b', 'E(w,b) = Σ(yi - (wxi + b))²', 'w* = (XᵀX)⁻¹XᵀY'],
                'concepts': ['Least Squares', 'Weight estimation', 'Bias term', 'Pseudoinverse'],
                'applications': ['Regression analysis', 'Prediction tasks']
            },
            'Classification': {
                'keywords': ['Classification', 'logistic', 'decision boundary', 'class'],
                'key_equations': ['P(C|x) = 1/(1 + e^(-w·x+b))', 'decision: a(x) = 0'],
                'concepts': ['Binary classification', 'Logistic function', 'Decision boundary', 'Class probability'],
                'applications': ['Email spam filtering', 'Image recognition', 'Medical diagnosis']
            },
            'Probability Theory': {
                'keywords': ['Probability', 'Bayes', 'distribution', 'PDF', 'Gaussian'],
                'key_equations': ['P(A,B) = P(A|B)P(B)', 'P(A|B) = P(B|A)P(A)/P(B)', 'G(x|μ,σ) = (1/√(2πσ²))exp(-(x-μ)²/(2σ²))'],
                'concepts': ['Joint probability', 'Conditional probability', 'Bayes Rule', 'Gaussian distribution', 'PDF'],
                'applications': ['Inference', 'Parameter estimation']
            },
            'Clustering': {
                'keywords': ['Clustering', 'K-means', 'Gaussian', 'mixture'],
                'key_equations': ['μ_k = (Σᵢ zᵢₖ xᵢ)/(Σᵢ zᵢₖ)', 'E = Σᵢ Σₖ zᵢₖ ||xᵢ - μₖ||²'],
                'concepts': ['K-means clustering', 'Mixtures of Gaussians', 'Cluster assignment', 'Centroid'],
                'applications': ['Customer segmentation', 'Image clustering', 'Data discovery']
            },
            'Neural Networks': {
                'keywords': ['Neural', 'sigmoid', 'weights', 'activation'],
                'key_equations': ['g(a) = 1/(1 + e^(-a))', 'y = Σⱼ wⱼ g(wⱼ⁽¹⁾x + bⱼ) + b'],
                'concepts': ['Sigmoid function', 'Nonlinear transformation', 'Hidden units', 'Weight decay'],
                'applications': ['Complex pattern learning', 'Non-linear regression']
            },
            'PCA & Dimensionality Reduction': {
                'keywords': ['PCA', 'Principal Components', 'dimensionality', 'eigenvalue'],
                'key_equations': ['y = Wx + b', 'K = (1/N)Σ(yᵢ - ȳ)(yᵢ - ȳ)ᵀ', 'x = W^T(y - b)'],
                'concepts': ['Eigendecomposition', 'Variance maximization', 'Data compression', 'Whitening'],
                'applications': ['Data visualization', 'Feature extraction', 'Noise reduction']
            },
            'Bayesian Methods': {
                'keywords': ['Bayesian', 'posterior', 'prior', 'likelihood'],
                'key_equations': ['P(w|D) = P(D|w)P(w)/P(D)', 'P(y_new|D) = ∫P(y_new|w)P(w|D)dw'],
                'concepts': ['Posterior distribution', 'Prior belief', 'Likelihood', 'Model uncertainty'],
                'applications': ['Parameter inference', 'Uncertainty quantification']
            },
            'Optimization': {
                'keywords': ['Gradient Descent', 'optimization', 'convergence', 'line search'],
                'key_equations': ['w_{t+1} = w_t - λ∇E(w_t)', '∇E = (dE/dw₁, ..., dE/dwₙ)ᵀ'],
                'concepts': ['Gradient descent', 'Local minima', 'Step size', 'Convergence'],
                'applications': ['Parameter learning', 'Model training']
            },
            'Cross Validation': {
                'keywords': ['Cross Validation', 'validation set', 'generalization'],
                'key_equations': ['Error = Σᵢ ||yᵢ - f(xᵢ)||²'],
                'concepts': ['Overfitting', 'Underfitting', 'Model selection', 'K-fold'],
                'applications': ['Hyperparameter tuning', 'Model comparison']
            },
        }
        
        return topics
    
    def generate_detailed_report(self, output_file):
        """Generate comprehensive detailed report"""
        report = []
        
        report.append("="*100)
        report.append("MACHINE LEARNING NOTES: COMPREHENSIVE TOPIC ANALYSIS")
        report.append("="*100)
        report.append("")
        
        topics = self.analyze_topics()
        
        for topic_name, topic_info in topics.items():
            report.append(f"\n{'═'*100}")
            report.append(f"TOPIC: {topic_name.upper()}")
            report.append(f"{'═'*100}\n")
            
            report.append(f"KEY CONCEPTS:")
            for concept in topic_info['concepts']:
                report.append(f"  • {concept}")
            
            report.append(f"\nKEY EQUATIONS:")
            for eq in topic_info['key_equations']:
                report.append(f"  • {eq}")
            
            report.append(f"\nAPPLICATIONS:")
            for app in topic_info['applications']:
                report.append(f"  • {app}")
            
            report.append("")
        
        report.append("\n" + "="*100)
        report.append("MATHEMATICAL EQUATIONS & FORMULAS SUMMARY")
        report.append("="*100)
        
        topics = self.analyze_topics()
        all_equations = []
        for topic_data in topics.values():
            all_equations.extend(topic_data['key_equations'])
        
        for i, eq in enumerate(all_equations, 1):
            report.append(f"\n{i}. {eq}")
        
        # Write report
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write('\n'.join(report))
        
        return '\n'.join(report)


# Run analysis
if __name__ == "__main__":
    json_file = r"c:\Users\charv\OneDrive - WOXSEN UNIVERSITY\anlp_rag\ML_NOTES_SEGREGATED.json"
    
    analyzer = DetailedMLAnalyzer(json_file)
    
    output_file = r"c:\Users\charv\OneDrive - WOXSEN UNIVERSITY\anlp_rag\ML_DETAILED_ANALYSIS.txt"
    report = analyzer.generate_detailed_report(output_file)
    
    print("✓ Detailed analysis saved!")
    print(f"✓ Output file: {output_file}")
