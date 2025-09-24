# 🧬 Integrated Species Classification System

A comprehensive AI-driven system that combines deep learning models for known species identification with unsupervised learning for novel species discovery, complete with interactive visualizations and biodiversity analysis.

## 🌟 Key Features

### 🔍 Interactive Species Analysis
- **User Input Processing**: Manual input, file upload (FASTQ), or sample data
- **Real-time Analysis**: Instant results with comprehensive reporting
- **Multi-format Support**: Handles various DNA sequence formats

### 🤖 Deep Learning for Known Species
- **CNN Model**: Convolutional Neural Network for sequence pattern recognition
- **RNN Model**: Recurrent Neural Network for sequential data analysis
- **Ensemble Approach**: Combines both models for improved accuracy
- **Confidence Scoring**: Provides confidence levels for each prediction

### 🆕 Unsupervised Learning for Novel Species
- **DBSCAN Clustering**: Discovers novel species clusters
- **Novelty Scoring**: Quantitative assessment of species novelty
- **Cluster Analysis**: Detailed analysis of novel species groups
- **Storage System**: Saves novel species for future comparisons

### 🔬 Species Comparison & Family Analysis
- **Taxonomic Similarity**: Calculates similarity across taxonomic levels
- **Sequence Similarity**: Compares DNA sequences using k-mer analysis
- **Family Relationships**: Analyzes relationships within taxonomic families
- **Phylogenetic Distance**: Measures evolutionary distance between species

### 🌿 Biodiversity Metrics
- **Basic Metrics**: Species richness, total individuals, known/novel species counts
- **Advanced Metrics**: Shannon diversity, Simpson diversity, evenness, dominance
- **Taxonomic Diversity**: Diversity analysis across taxonomic levels
- **Ecological Metrics**: Beta diversity, turnover rate, nestedness, functional diversity

### 📊 Interactive Visualizations
- **Species Abundance Charts**: Pie charts and bar plots
- **Biodiversity Dashboards**: Comprehensive biodiversity metrics
- **Novel Species Discovery**: Visual representation of novel clusters
- **Confidence Distributions**: Classification confidence analysis
- **Phylogenetic Trees**: Taxonomic hierarchy visualization

## 🚀 Quick Start

### 1. Installation

```bash
# Install required dependencies
pip install -r requirements.txt

# Or install individual packages
pip install streamlit plotly scikit-learn torch pandas numpy matplotlib seaborn
```

### 2. Launch the Integrated Dashboard

```bash
# Navigate to scripts directory
cd scripts

# Launch the integrated dashboard
python launch_integrated_dashboard.py
```

The dashboard will open in your browser at `http://localhost:8501`

### 3. Using the System

1. **Select Analysis Mode**: Choose from the navigation menu
2. **Input DNA Sequences**: Use manual input, file upload, or sample data
3. **Run Analysis**: Click "Run Complete Analysis" to process sequences
4. **View Results**: Explore results across different analysis modes
5. **Download Reports**: Save comprehensive analysis reports

## 📋 System Components

### 🧬 Interactive Species Classifier (`interactive_species_classifier.py`)
- Main classification system
- Handles both known and unknown species
- Deep learning model training and inference
- Novel species discovery and clustering

### 🔬 Species Comparison Analyzer (`species_comparison_analyzer.py`)
- Species-to-species comparison
- Family relationship analysis
- Taxonomic similarity calculations
- Phylogenetic distance measurements

### 🌿 Biodiversity Metrics Calculator (`biodiversity_metrics_calculator.py`)
- Comprehensive biodiversity calculations
- Advanced ecological metrics
- Taxonomic diversity analysis
- Interactive visualizations

### 📊 Integrated Dashboard (`integrated_species_dashboard.py`)
- Unified interface for all features
- Real-time analysis and reporting
- Interactive visualizations
- Comprehensive result display

## 🔧 Analysis Modes

### 1. 🧬 Interactive Analysis
- **Purpose**: Complete species analysis pipeline
- **Input**: DNA sequences (manual, file, or sample)
- **Output**: Comprehensive analysis report
- **Features**: Known species classification, novel species discovery, biodiversity metrics

### 2. 🔍 Known Species Analysis
- **Purpose**: Analyze known species using trained models
- **Input**: DNA sequences
- **Output**: Species predictions with confidence scores
- **Features**: CNN/RNN ensemble classification, confidence analysis

### 3. 🆕 Novel Species Discovery
- **Purpose**: Discover and analyze novel species
- **Input**: DNA sequences
- **Output**: Novel species clusters with novelty scores
- **Features**: DBSCAN clustering, novelty scoring, cluster analysis

### 4. 🔬 Species Comparison
- **Purpose**: Compare species and analyze relationships
- **Input**: Species pairs or families
- **Output**: Similarity analysis and relationship insights
- **Features**: Taxonomic similarity, sequence comparison, family analysis

### 5. 🌿 Biodiversity Analysis
- **Purpose**: Calculate comprehensive biodiversity metrics
- **Input**: Species data
- **Output**: Biodiversity metrics and visualizations
- **Features**: Shannon/Simpson diversity, evenness, taxonomic diversity

### 6. 📊 Complete Report
- **Purpose**: Generate comprehensive analysis reports
- **Input**: Analysis results
- **Output**: Detailed reports with insights and recommendations
- **Features**: Biological insights, recommendations, downloadable reports

## 📊 Output Examples

### Known Species Classification
```
Species: Actinia equina
Confidence: 0.847
Model: ensemble_cnn_rnn
```

### Novel Species Discovery
```
Cluster ID: novel_cluster_1
Size: 15 sequences
Novelty Score: 0.723
Discovery Date: 2024-01-15
```

### Biodiversity Metrics
```
Species Richness: 12
Shannon Diversity: 2.456
Simpson Diversity: 0.823
Evenness: 0.789
```

## 🔬 Technical Details

### Deep Learning Models
- **CNN Architecture**: 1D convolutional layers with max pooling
- **RNN Architecture**: Bidirectional LSTM with attention
- **Training**: Adam optimizer with cross-entropy loss
- **Ensemble**: Weighted average of CNN and RNN predictions

### Unsupervised Learning
- **Clustering**: DBSCAN with adaptive parameters
- **Feature Extraction**: K-mer frequencies, GC content, sequence length
- **Novelty Scoring**: Distance-based novelty assessment
- **Storage**: JSON-based cluster storage system

### Biodiversity Calculations
- **Shannon Diversity**: H' = -Σ(pi × log2(pi))
- **Simpson Diversity**: 1 - Σ(pi²)
- **Evenness**: J' = H' / log2(S)
- **Chao1 Estimator**: S + (f1²) / (2 × f2)

## 📁 File Structure

```
scripts/
├── interactive_species_classifier.py      # Main classification system
├── species_comparison_analyzer.py         # Species comparison analysis
├── biodiversity_metrics_calculator.py     # Biodiversity calculations
├── integrated_species_dashboard.py        # Unified dashboard
├── launch_integrated_dashboard.py         # Dashboard launcher
└── requirements.txt                       # Dependencies

results/
├── taxonomic_database.json               # Species database
├── novel_clusters.json                   # Novel species clusters
├── taxonomic_cnn.pth                     # Trained CNN model
├── taxonomic_rnn.pth                     # Trained RNN model
└── label_encoders.pkl                    # Label encoders

data/
├── SRR11851935_1.fastq                   # Sample FASTQ files
├── SRR11851935_2.fastq
├── SRR12076396_1.fastq
└── SRR12076396_2.fastq
```

## 🎯 Use Cases

### 1. Environmental DNA (eDNA) Analysis
- Analyze environmental samples for species presence
- Monitor biodiversity in ecosystems
- Detect invasive or endangered species

### 2. Taxonomic Research
- Identify unknown species in samples
- Compare species relationships
- Analyze evolutionary patterns

### 3. Conservation Biology
- Assess ecosystem health
- Monitor species diversity
- Track conservation efforts

### 4. Marine Biology
- Analyze marine samples
- Study coral reef biodiversity
- Monitor ocean ecosystem health

## 🔧 Customization

### Adding New Species
```python
# Add to taxonomic database
species_data = {
    'sequences': ['ATCGATCG...', 'GCTAGCTA...'],
    'taxonomy': {
        'Kingdom': 'Eukaryota',
        'Phylum': 'Cnidaria',
        'Class': 'Anthozoa',
        'Order': 'Actiniaria',
        'Family': 'Actiniidae',
        'Genus': 'Actinia',
        'Species': 'Actinia equina'
    },
    'sequence_count': 2
}
```

### Modifying Models
```python
# Customize CNN architecture
class CustomCNN(nn.Module):
    def __init__(self, input_size=200, num_classes=10):
        super().__init__()
        # Add your custom layers here
        pass
```

### Adding New Metrics
```python
# Add custom biodiversity metric
def custom_diversity_metric(species_counts):
    # Your custom calculation
    return metric_value
```

## 🐛 Troubleshooting

### Common Issues

1. **Models not loading**
   - Ensure models are trained first
   - Check file paths in results directory

2. **Low classification accuracy**
   - Retrain models with more data
   - Adjust model hyperparameters

3. **No novel species found**
   - Check clustering parameters
   - Verify input sequence quality

4. **Dashboard not loading**
   - Install required dependencies
   - Check port availability (8501)

### Performance Optimization

1. **Large datasets**
   - Use batch processing
   - Implement data sampling
   - Optimize model architecture

2. **Memory issues**
   - Reduce batch sizes
   - Use data generators
   - Implement model checkpointing

## 📚 References

- [Deep Learning for DNA Sequence Analysis](https://example.com)
- [Biodiversity Metrics in Ecology](https://example.com)
- [Unsupervised Learning for Novel Species Discovery](https://example.com)
- [Taxonomic Classification Methods](https://example.com)

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- Streamlit team for the excellent dashboard framework
- PyTorch team for deep learning capabilities
- Scikit-learn team for machine learning tools
- Plotly team for interactive visualizations

## 📞 Support

For questions, issues, or contributions:
- Create an issue on GitHub
- Contact the development team
- Check the documentation

---

**🧬 Integrated Species Classification System - Powered by AI**
