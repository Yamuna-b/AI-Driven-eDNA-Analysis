# eDNA Species Predictor Dashboard

A comprehensive Streamlit dashboard for predicting species from DNA sequences using your trained eDNA model.

## Features

🧬 **DNA Sequence Analysis**
- Input DNA sequences manually, upload FASTA files, or use example sequences
- Validates DNA sequences (A, T, C, G only)
- Calculates sequence statistics (length, GC content, base composition)

🔬 **Species Prediction**
- Uses your trained `edna_model.pkl` for predictions
- Returns **Known Species** for sequences matching your training data
- Returns **Unknown** for potentially novel species
- Provides confidence scores and detailed explanations

📊 **Comprehensive Visualizations**
- Species distribution pie charts
- Confidence score histograms
- GC content vs confidence scatter plots
- Batch analysis for multiple sequences

🤖 **Model Integration**
- Loads your trained model from `models/edna_model.pkl`
- Real-time predictions with confidence scoring
- Handles both single sequences and batch processing

## Installation

1. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

2. **Verify Model Location**
   Ensure your trained model is located at:
   ```
   models/edna_model.pkl
   ```

## Usage

### Running the Dashboard

```bash
cd scripts
streamlit run edna_species_predictor.py
```

The dashboard will open in your web browser at `http://localhost:8501`

### Dashboard Sections

#### 1. Species Prediction
- **Manual Entry**: Type or paste DNA sequences directly
- **File Upload**: Upload FASTA files with multiple sequences
- **Example Sequences**: Use pre-loaded example sequences for testing

#### 2. Batch Analysis
- Process multiple sequences simultaneously
- View aggregate statistics and visualizations
- Export results for further analysis

#### 3. Model Information
- View model details and performance metrics
- Understand prediction criteria and confidence levels
- Check species database information

#### 4. Help & Documentation
- Detailed usage instructions
- Sequence requirements and formatting
- Confidence level explanations

### Input Requirements

**Valid DNA Sequences:**
- Only use bases: A, T, C, G
- Minimum length: 50 bases (recommended for reliable predictions)
- Remove ambiguous bases (N, R, Y, etc.)

**Supported Formats:**
- Plain text sequences
- FASTA format files
- Multi-sequence FASTA files

### Prediction Results

The dashboard provides three types of predictions:

- **✅ Known Species** (Confidence > 70%): Strong match with training data
- **⚠️ Uncertain** (Confidence 50-70%): Possible match, requires verification  
- **❌ Unknown** (Confidence < 50%): Likely novel or unknown species

### Example Usage

1. **Single Sequence Prediction:**
   - Enter: `ATCGATCGATCGATCGATCGATCG`
   - Get: Species prediction with confidence score

2. **Batch Processing:**
   - Upload a FASTA file with multiple sequences
   - View comprehensive analysis with visualizations

3. **Novel Species Discovery:**
   - Input unknown sequences
   - Identify potentially novel species with low confidence scores

## Model Details

Your trained model (`edna_model.pkl`) is used for:
- DNA sequence feature extraction
- Species classification
- Confidence score calculation
- Novel species detection

The dashboard automatically loads your model and uses it for real-time predictions.

## Troubleshooting

**Model Not Found:**
- Ensure `edna_model.pkl` is in the `models/` folder
- Check file permissions and path

**Invalid Sequence Error:**
- Use only A, T, C, G bases
- Remove spaces and special characters
- Ensure minimum sequence length

**Performance Issues:**
- For large batch processing, consider processing in smaller chunks
- Ensure sufficient system memory for model loading

## Customization

You can customize the dashboard by modifying:
- `predict_species()` function for your specific model implementation
- Species database in the mock prediction logic
- Confidence thresholds and scoring criteria
- Visualization styles and layouts

## File Structure

```
AI Driven EDNA/
├── models/
│   └── edna_model.pkl          # Your trained model
├── scripts/
│   ├── edna_species_predictor.py  # Main dashboard
│   └── eDNA_dashboard.py       # Original dashboard
├── requirements.txt            # Dependencies
└── README_SPECIES_PREDICTOR.md # This file
```

## Next Steps

1. **Run the Dashboard**: Start with the example sequences to test functionality
2. **Upload Your Data**: Use your own DNA sequences for prediction
3. **Analyze Results**: Review predictions and confidence scores
4. **Batch Processing**: Upload FASTA files for large-scale analysis
5. **Export Results**: Download prediction results for further analysis

## Support

For issues or questions:
1. Check the Help & Documentation section in the dashboard
2. Verify model file location and format
3. Ensure all dependencies are installed correctly
4. Review sequence input requirements

---

**Built with Streamlit | Powered by Your Trained eDNA Model**
