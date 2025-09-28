# Sentiment Analysis Project

A comprehensive machine learning project for sentiment analysis using both traditional ML approaches and modern transformer models.

## 🎯 Project Overview

This project implements multiple sentiment analysis approaches:
- **Traditional ML Models**: Logistic Regression, SVM, Naive Bayes, Random Forest
- **Transformer Models**: Pre-trained models from Hugging Face
- **Text Preprocessing**: Comprehensive text cleaning and feature extraction
- **Model Comparison**: Side-by-side evaluation of different approaches

## 📁 Project Structure

```
sentiment_analysis_project/
├── data/                   # Data storage directory
├── models/                 # Saved model files
├── notebooks/              # Jupyter notebooks for exploration
│   └── 01_explore_and_train.ipynb
├── src/                    # Source code
│   ├── preprocessing.py    # Text preprocessing utilities
│   └── sentiment_analyzer.py  # Main ML models and analysis
├── tests/                  # Unit tests
├── requirements.txt        # Project dependencies
├── config.yaml            # Configuration settings
└── README.md              # This file
```

## 🚀 Quick Start

### 1. Environment Setup

```powershell
# Create virtual environment
python -m venv .venv

# Activate virtual environment (Windows PowerShell)
.\.venv\Scripts\Activate.ps1

# Install dependencies
pip install -r requirements.txt

# Download NLTK data (will be done automatically on first run)
python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords'); nltk.download('wordnet')"
```

### 2. Run the Demo

```powershell
# Run the main demo script
cd src
python sentiment_analyzer.py
```

### 3. Use Jupyter Notebook

```powershell
# Start Jupyter
jupyter lab

# Open notebooks/01_explore_and_train.ipynb
```

### 4. Test Voice Input (NEW! 🎤)

```powershell
# Test voice functionality
python voice_demo.py

# Or run the GUI with voice support
python launch_gui.py
```

#### 🎤 Voice Input Requirements

- **Microphone**: Working microphone connected to your system
- **Internet Connection**: Required for Google Speech Recognition API
- **Audio Dependencies**: PyAudio, SpeechRecognition, pydub (installed automatically)
- **Supported Formats**: WAV, MP3, M4A for file analysis

## 💻 Usage Examples

### Basic Usage

```python
from src.sentiment_analyzer import TraditionalSentimentAnalyzer, TransformerSentimentAnalyzer
from src.preprocessing import load_sample_data, create_train_test_split

# Load sample data
df = load_sample_data()
X_train, X_test, y_train, y_test = create_train_test_split(df, 'text', 'sentiment')

# Train traditional model
model = TraditionalSentimentAnalyzer(model_type='logistic_regression')
model.train(X_train, y_train)

# Make predictions
texts = ["I love this product!", "This is terrible."]
predictions = model.predict(texts)
print(predictions)  # ['positive', 'negative']

# Use transformer model
transformer = TransformerSentimentAnalyzer()
transformer_predictions = transformer.predict(texts)
print(transformer_predictions)
```

### Model Comparison

```python
from src.sentiment_analyzer import SentimentAnalysisComparator

# Compare multiple models
comparator = SentimentAnalysisComparator()
comparator.add_traditional_model("Logistic Regression", "logistic_regression")
comparator.add_traditional_model("SVM", "svm")
comparator.add_transformer_model("Transformer", "cardiffnlp/twitter-roberta-base-sentiment-latest")

results = comparator.compare_models(X_train, y_train, X_test, y_test)
print(results)
```

### Voice Input Analysis (NEW! 🎤)

```python
from src.voice_processor import VoiceInputProcessor, VoiceSentimentAnalyzer
from src.sentiment_analyzer import TransformerSentimentAnalyzer

# Test microphone setup
processor = VoiceInputProcessor()
test_result = processor.test_microphone(duration=5)
print(f"Microphone test: {test_result}")

# Real-time voice sentiment analysis
transformer_model = TransformerSentimentAnalyzer()
voice_analyzer = VoiceSentimentAnalyzer(transformer_model)

# Set up callbacks for real-time feedback
def on_text_recognized(text):
    print(f"Heard: {text}")

def on_sentiment_detected(sentiment_data):
    sentiment = sentiment_data['sentiment']
    confidence = sentiment_data['confidence']
    print(f"Sentiment: {sentiment} (confidence: {confidence:.2f})")

# Start real-time analysis
voice_analyzer.start_real_time_analysis(
    on_sentiment_callback=on_sentiment_detected,
    on_text_callback=on_text_recognized
)

# Stop and get summary
summary = voice_analyzer.stop_real_time_analysis()
print(f"Overall sentiment: {summary['overall_sentiment']}")
print(f"Total phrases: {summary['total_phrases']}")

# Analyze audio file
result = voice_analyzer.analyze_audio_file('path/to/audio.wav')
if result['success']:
    print(f"File text: {result['text']}")
    print(f"Sentiment: {result['primary_sentiment']} ({result['confidence']:.2f})")
```

## 🔧 Configuration

Edit `config.yaml` to customize:
- Model parameters
- Preprocessing options
- File paths
- API settings

## 📊 Features

- **Multiple Model Types**: Traditional ML and modern transformers
- **Comprehensive Preprocessing**: Text cleaning, tokenization, stemming/lemmatization
- **Model Evaluation**: Accuracy, precision, recall, F1-score
- **Visualization**: Model comparison plots
- **🎤 Voice Input Support**: Real-time speech recognition and sentiment analysis
- **🔊 Audio File Processing**: Batch analysis of audio files
- **Real-time Feedback**: Live sentiment analysis during voice input
- **Extensible**: Easy to add new models and features
- **Production Ready**: Model saving/loading, API deployment ready

## 🚀 API Endpoints

The Flask server provides REST API endpoints for both text and voice analysis:

### Text Analysis Endpoints
- `POST /api/analyze` - Analyze sentiment for single text
- `POST /api/batch_analyze` - Analyze sentiment for multiple texts
- `GET /api/examples` - Get sample texts for testing
- `GET /api/model_status` - Check model loading status

### Voice Analysis Endpoints (NEW! 🎤)
- `GET /api/voice/test_microphone` - Test microphone functionality
- `GET /api/voice/devices` - List available audio devices
- `POST /api/voice/start_recording` - Start real-time voice recording
- `POST /api/voice/stop_recording` - Stop recording and get results
- `GET /api/voice/status` - Get current recording status
- `POST /api/voice/analyze_file` - Analyze uploaded audio file

### Example API Usage

```python
import requests

# Test microphone
response = requests.get('http://localhost:5000/api/voice/test_microphone')
print(response.json())

# Start voice recording
start_data = {'session_id': 'my_session', 'model_type': 'transformer'}
response = requests.post('http://localhost:5000/api/voice/start_recording', json=start_data)
print(response.json())

# Stop recording and get results
stop_data = {'session_id': 'my_session'}
response = requests.post('http://localhost:5000/api/voice/stop_recording', json=stop_data)
result = response.json()
print(f"Overall sentiment: {result['summary']['overall_sentiment']}")
```

## 🧪 Testing

```powershell
# Run tests (when implemented)
python -m pytest tests/
```

## 📈 Model Performance

The project includes several pre-configured models:

| Model Type | Typical Accuracy | Training Time | Prediction Speed |
|------------|-----------------|---------------|------------------|
| Logistic Regression | 85-90% | Fast | Very Fast |
| SVM | 80-85% | Moderate | Fast |
| Naive Bayes | 75-80% | Very Fast | Very Fast |
| Transformer | 90-95% | N/A (pre-trained) | Moderate |

## 🔮 Future Enhancements

- [ ] Real-time sentiment analysis API
- [ ] Support for multiple languages
- [ ] Advanced preprocessing options
- [ ] Model ensemble methods
- [ ] Docker containerization
- [ ] MLOps pipeline integration
- [ ] Custom dataset support
- [ ] Aspect-based sentiment analysis

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Submit a pull request

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- Hugging Face for transformer models
- scikit-learn for traditional ML algorithms
- NLTK for text processing utilities
- The open-source community for inspiration and tools

---

**Happy Analyzing! 🎉**
