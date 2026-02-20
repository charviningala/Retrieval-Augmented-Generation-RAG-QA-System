import json
import re
from collections import defaultdict
from pathlib import Path

class MLNotesProcessor:
    """Process ML notes to extract topics, summaries, and equations"""
    
    def __init__(self, json_file):
        self.json_file = json_file
        self.data = self._load_json()
        self.topics = {}
        self.equations = []
        self.summaries = {}
        
    def _load_json(self):
        with open(self.json_file, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    def _extract_equations(self, text):
        """Extract mathematical equations from text"""
        equations = []
        
        # Pattern for equations with = sign
        eq_patterns = [
            r'([a-zA-Z0-9_\s\(\)\[\]\/\*\+\-\.]+\s*=\s*[a-zA-Z0-9_\s\(\)\[\]\/\*\+\-\.,\"\']+)',
            r'([\w\s]+\([\w\s\,\.\-\+\*\/\(\)]+\))',
            r'\([\d\w\s\+\-\*\/\.]+\)',
        ]
        
        for pattern in eq_patterns:
            matches = re.findall(pattern, text)
            equations.extend(matches)
        
        return equations
    
    def _extract_topic(self, text):
        """Extract main topic from text"""
        lines = text.split('\n')
        for line in lines:
            if re.match(r'^[A-Z][A-Za-z\s]+$', line.strip()):
                return line.strip()
        return None
    
    def _summarize_text(self, text, max_sentences=3):
        """Create a brief summary of text"""
        # Split by sentences
        sentences = re.split(r'[.!?]+', text)
        sentences = [s.strip() for s in sentences if len(s.strip()) > 20]
        
        if not sentences:
            return text[:200]
        
        # Get first few sentences
        summary = '. '.join(sentences[:max_sentences])
        if not summary.endswith('.'):
            summary += '.'
        return summary[:500]
    
    def segregate_by_topic(self):
        """Segregate notes by topics"""
        
        topic_map = {
            # Introduction
            'introduction': ['Introduction to Machine Learning', 'Types of Machine Learning', 'INTRODUCTION', 'MACHINE LEARNING'],
            
            # Regression
            'regression': ['Linear Regression', 'Nonlinear Regression', '2 Linear', '3 Nonlinear'],
            
            # Probability Theory
            'probability': ['Basic Probability Theory', 'Probability Density Functions', 'Discrete random', '5 Basic', '6 Probability'],
            
            # Estimation
            'estimation': ['Estimation', 'Parameter estimation', '7 Estimation', 'Bayes'],
            
            # Classification
            'classification': ['Classification', 'Logistic Regression', 'K-Nearest', 'Naive Bayes', '8 Classification'],
            
            # Optimization
            'optimization': ['Gradient Descent', 'Quadratics', '4 Quadratics', '9 Gradient'],
            
            # Cross Validation
            'validation': ['Cross Validation', '10 Cross'],
            
            # Bayesian Methods
            'bayesian': ['Bayesian Methods', 'Bayesian Regression', '11 Bayesian'],
            
            # Monte Carlo
            'monte_carlo': ['Monte Carlo', '12 Monte'],
            
            # Dimensionality Reduction
            'dim_reduction': ['Principal Components Analysis', 'PCA', '13 Principal'],
            
            # Clustering
            'clustering': ['Clustering', 'K-means', 'Mixtures', '15 Clustering'],
            
            # Advanced Methods
            'advanced': ['Hidden Markov', 'Support Vector', 'AdaBoost', 'Lagrange', '14 Lagrange', '16 Hidden', '17 Support', '18 AdaBoost'],
        }
        
        for entry in self.data:
            content = entry.get('content', '')
            page = entry.get('page_number', 'N/A')
            pdf = entry.get('pdf_source', 'unknown')
            
            # Find matching topic
            matched_topic = 'other'
            for topic, keywords in topic_map.items():
                if any(keyword.lower() in content.lower() for keyword in keywords):
                    matched_topic = topic
                    break
            
            # Store by topic
            if matched_topic not in self.topics:
                self.topics[matched_topic] = []
            
            self.topics[matched_topic].append({
                'content': content,
                'page_number': page,
                'pdf_source': pdf
            })
        
        return self.topics
    
    def extract_all_equations(self):
        """Extract equations from all content"""
        for entry in self.data:
            content = entry.get('content', '')
            eqs = self._extract_equations(content)
            for eq in eqs:
                if len(eq.strip()) > 3:
                    self.equations.append({
                        'equation': eq.strip(),
                        'page_number': entry.get('page_number', 'N/A'),
                        'context': content[:100]
                    })
        
        # Remove duplicates
        unique_eqs = {e['equation']: e for e in self.equations}
        self.equations = list(unique_eqs.values())
        
        return self.equations
    
    def create_summaries(self):
        """Create summaries for each topic"""
        for topic, entries in self.topics.items():
            combined_text = '\n'.join([e['content'] for e in entries])
            summary = self._summarize_text(combined_text, max_sentences=2)
            self.summaries[topic] = summary
        
        return self.summaries
    
    def generate_report(self, output_file):
        """Generate comprehensive report"""
        report = []
        
        report.append("="*80)
        report.append("MACHINE LEARNING NOTES: SEGREGATED & SUMMARIZED")
        report.append("="*80)
        report.append("")
        
        # Overall statistics
        report.append("DOCUMENT STATISTICS")
        report.append("-" * 80)
        pages = [e.get('page_number', 0) for e in self.data if isinstance(e.get('page_number'), int)]
        max_page = max(pages) if pages else 0
        report.append(f"Total Pages: {max_page}")
        report.append(f"Total Topics Identified: {len(self.topics)}")
        report.append(f"Total Equations Found: {len(self.equations)}")
        report.append("")
        
        # Topic summaries
        report.append("TOPIC SUMMARIES")
        report.append("-" * 80)
        for topic in sorted(self.topics.keys()):
            report.append("")
            report.append(f"\n### {topic.upper().replace('_', ' ')} ###")
            report.append(f"Entries: {len(self.topics[topic])}")
            
            # Get pages covered
            pages = [e.get('page_number', 0) for e in self.topics[topic] if isinstance(e.get('page_number'), int)]
            if pages:
                report.append(f"Page Range: {min(pages)} - {max(pages)}")
            
            # Summary
            if topic in self.summaries:
                report.append(f"\nSummary: {self.summaries[topic]}")
            
            # Sample content preview
            if self.topics[topic]:
                first_content = self.topics[topic][0]['content'][:150]
                report.append(f"\nPreview: {first_content}...")
        
        report.append("\n" + "="*80)
        report.append("EQUATIONS IDENTIFIED")
        report.append("="*80)
        report.append("")
        
        for i, eq_data in enumerate(sorted(self.equations, key=lambda x: str(x.get('page_number', 0)))[:50], 1):
            report.append(f"{i}. {eq_data['equation']}")
            report.append(f"   Page: {eq_data['page_number']}")
            report.append("")
        
        if len(self.equations) > 50:
            report.append(f"... and {len(self.equations) - 50} more equations")
        
        # Write report
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write('\n'.join(report))
        
        return '\n'.join(report)


def process_notes():
    """Main processing function"""
    json_path = r"c:\Users\charv\OneDrive - WOXSEN UNIVERSITY\anlp_rag\STRUCTURED_ML.json"
    output_path = r"c:\Users\charv\OneDrive - WOXSEN UNIVERSITY\anlp_rag\ML_NOTES_ANALYZED.txt"
    output_json = r"c:\Users\charv\OneDrive - WOXSEN UNIVERSITY\anlp_rag\ML_NOTES_SEGREGATED.json"
    
    print("Processing ML Notes...")
    
    processor = MLNotesProcessor(json_path)
    
    print("1. Segregating by topics...")
    processor.segregate_by_topic()
    
    print("2. Extracting equations...")
    processor.extract_all_equations()
    
    print("3. Creating summaries...")
    processor.create_summaries()
    
    print("4. Generating report...")
    report = processor.generate_report(output_path)
    
    print(f"\n✓ Report saved to: {output_path}")
    print(f"\nTopics Found: {len(processor.topics)}")
    print(f"Equations Extracted: {len(processor.equations)}")
    
    # Save structured data as JSON
    output_data = {
        'topics': {
            topic: {
                'count': len(entries),
                'summary': processor.summaries.get(topic, ''),
                'pages': list(set([str(e.get('page_number', '')) for e in entries]))
            }
            for topic, entries in processor.topics.items()
        },
        'equations': [
            {
                'equation': eq['equation'],
                'page': str(eq.get('page_number', '')),
                'context': eq['context']
            }
            for eq in processor.equations[:20]
        ],
        'total_equations': len(processor.equations)
    }
    
    with open(output_json, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=2)
    
    print(f"✓ Structured data saved to: {output_json}")
    
    return processor


if __name__ == "__main__":
    processor = process_notes()
    
    # Print summary
    print("\n" + "="*80)
    print("TOPIC DISTRIBUTION")
    print("="*80)
    for topic, entries in sorted(processor.topics.items()):
        print(f"{topic.upper():30} : {len(entries):3} pages")
