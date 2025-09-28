"""
Flask API server for the Sentiment Analysis GUI
Provides REST endpoints for the web interface
"""

import sys
import os
sys.path.append('src')

from flask import Flask, request, jsonify, render_template, send_from_directory
from flask_cors import CORS
import json
import traceback
from sentiment_analyzer import TransformerSentimentAnalyzer, TraditionalSentimentAnalyzer
from preprocessing import TextPreprocessor
from voice_processor import VoiceInputProcessor, VoiceSentimentAnalyzer
import threading
import time
import base64
import tempfile
import io

app = Flask(__name__, 
           template_folder='gui/templates',
           static_folder='gui/static')
CORS(app)

# Global variables for models
transformer_model = None
traditional_model = None
preprocessor = TextPreprocessor()
voice_processor = None
voice_analyzer = None
active_voice_sessions = {}

def load_models():
    """Load models in background to avoid startup delay"""
    global transformer_model, traditional_model
    try:
        print("Loading transformer model...")
        transformer_model = TransformerSentimentAnalyzer()
        print("Transformer model loaded successfully!")
        
        print("Loading traditional model...")
        traditional_model = TraditionalSentimentAnalyzer()
        print("Traditional model loaded successfully!")
    except Exception as e:
        print(f"Error loading models: {e}")

# Load models in background thread
model_thread = threading.Thread(target=load_models)
model_thread.daemon = True
model_thread.start()

@app.route('/')
def index():
    """Serve the main GUI page"""
    return render_template('index.html')

@app.route('/api/analyze', methods=['POST'])
def analyze_sentiment():
    """Analyze sentiment for given text"""
    try:
        data = request.get_json()
        text = data.get('text', '').strip()
        model_type = data.get('model_type', 'transformer').lower()
        
        if not text:
            return jsonify({
                'error': 'No text provided',
                'status': 'error'
            }), 400
        
        # Check if models are loaded
        if transformer_model is None:
            return jsonify({
                'error': 'Models are still loading, please wait...',
                'status': 'loading'
            }), 503
        
        result = {}
        
        if model_type == 'transformer' and transformer_model:
            # Use transformer model
            predictions = transformer_model.predict_with_scores([text])
            pred_data = predictions[0]
            
            result = {
                'text': text,
                'model_type': 'transformer',
                'predictions': pred_data,
                'primary_sentiment': pred_data[0]['label'],
                'confidence': pred_data[0]['score'],
                'status': 'success'
            }
            
        elif model_type == 'traditional' and traditional_model:
            # For traditional model, we need to train it first with sample data
            from preprocessing import load_sample_data, create_train_test_split
            
            # Quick training if not already trained
            if traditional_model.pipeline is None:
                df = load_sample_data()
                df_processed = preprocessor.preprocess_dataframe(df, 'text', 'sentiment')
                X_train, _, y_train, _ = create_train_test_split(
                    df_processed, 'text_processed', 'sentiment', test_size=0.2
                )
                traditional_model.train(X_train, y_train)
            
            prediction = traditional_model.predict([text])[0]
            probabilities = traditional_model.predict_proba([text])[0]
            
            # Convert to similar format as transformer
            labels = list(traditional_model.label_mapping.keys())
            pred_data = [
                {'label': labels[i], 'score': float(prob)}
                for i, prob in enumerate(probabilities)
            ]
            pred_data.sort(key=lambda x: x['score'], reverse=True)
            
            result = {
                'text': text,
                'model_type': 'traditional',
                'predictions': pred_data,
                'primary_sentiment': prediction,
                'confidence': max(probabilities),
                'status': 'success'
            }
        
        else:
            return jsonify({
                'error': f'Invalid model type: {model_type}',
                'status': 'error'
            }), 400
        
        return jsonify(result)
        
    except Exception as e:
        print(f"Error in analyze_sentiment: {str(e)}")
        traceback.print_exc()
        return jsonify({
            'error': str(e),
            'status': 'error'
        }), 500

@app.route('/api/batch_analyze', methods=['POST'])
def batch_analyze():
    """Analyze sentiment for multiple texts"""
    try:
        data = request.get_json()
        texts = data.get('texts', [])
        model_type = data.get('model_type', 'transformer').lower()
        
        if not texts or not isinstance(texts, list):
            return jsonify({
                'error': 'No texts provided or invalid format',
                'status': 'error'
            }), 400
        
        if len(texts) > 10:  # Limit batch size
            return jsonify({
                'error': 'Maximum 10 texts allowed per batch',
                'status': 'error'
            }), 400
        
        if transformer_model is None:
            return jsonify({
                'error': 'Models are still loading, please wait...',
                'status': 'loading'
            }), 503
        
        results = []
        
        for text in texts:
            if not text.strip():
                continue
                
            if model_type == 'transformer' and transformer_model:
                predictions = transformer_model.predict_with_scores([text])
                pred_data = predictions[0]
                
                results.append({
                    'text': text,
                    'primary_sentiment': pred_data[0]['label'],
                    'confidence': pred_data[0]['score'],
                    'predictions': pred_data
                })
        
        return jsonify({
            'results': results,
            'model_type': model_type,
            'status': 'success'
        })
        
    except Exception as e:
        print(f"Error in batch_analyze: {str(e)}")
        return jsonify({
            'error': str(e),
            'status': 'error'
        }), 500

@app.route('/api/model_status')
def model_status():
    """Check if models are loaded"""
    return jsonify({
        'transformer_loaded': transformer_model is not None,
        'traditional_loaded': traditional_model is not None,
        'status': 'ready' if transformer_model is not None else 'loading'
    })

@app.route('/api/examples')
def get_examples():
    """Get sample texts for testing"""
    examples = [
        "I absolutely love this product! It's amazing!",
        "This is terrible. I hate it so much.",
        "The service was okay, nothing special.",
        "Best purchase ever! Highly recommend!",
        "Disappointing quality. Expected better.",
        "It's fine, could be better but not bad.",
        "Exceptional experience! Will buy again!",
        "Poor value for money. Not worth it.",
        "Perfect! Exactly what I needed.",
        "Average product. Nothing impressive."
    ]
    
    return jsonify({
        'examples': examples,
        'status': 'success'
    })

# Voice Input API Endpoints

@app.route('/api/voice/test_microphone', methods=['GET'])
def test_microphone():
    """Test microphone functionality"""
    global voice_processor
    
    try:
        if voice_processor is None:
            voice_processor = VoiceInputProcessor()
        
        # Simple microphone test - just check if we can create the processor
        return jsonify({
            'success': True,
            'message': 'Microphone test successful',
            'status': 'success'
        })
    
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e),
            'status': 'error'
        })

@app.route('/api/voice/devices', methods=['GET'])
def get_audio_devices():
    """Get available audio input devices"""
    global voice_processor
    
    try:
        if voice_processor is None:
            voice_processor = VoiceInputProcessor()
        
        devices = voice_processor.get_audio_devices()
        return jsonify({
            'devices': devices,
            'status': 'success'
        })
    
    except Exception as e:
        return jsonify({
            'error': str(e),
            'status': 'error'
        })

@app.route('/api/voice/start_recording', methods=['POST'])
def start_voice_recording():
    """Start real-time voice recording and sentiment analysis"""
    global voice_analyzer, transformer_model, active_voice_sessions
    
    try:
        data = request.get_json()
        session_id = data.get('session_id', 'default')
        model_type = data.get('model_type', 'transformer')
        
        # Check if models are loaded
        if transformer_model is None:
            return jsonify({
                'error': 'Models are still loading, please wait...',
                'status': 'loading'
            }), 503
        
        # Clean up any existing session with same ID
        if session_id in active_voice_sessions:
            try:
                old_analyzer = active_voice_sessions[session_id]['analyzer']
                old_analyzer.stop_real_time_analysis()
                del active_voice_sessions[session_id]
            except Exception as cleanup_error:
                print(f"Error cleaning up old session: {cleanup_error}")
        
        # Choose model
        if model_type == 'transformer':
            selected_model = transformer_model
        else:
            # Ensure traditional model is trained
            if traditional_model and traditional_model.pipeline is None:
                from preprocessing import load_sample_data, create_train_test_split
                df = load_sample_data()
                df_processed = preprocessor.preprocess_dataframe(df, 'text', 'sentiment')
                X_train, _, y_train, _ = create_train_test_split(
                    df_processed, 'text_processed', 'sentiment', test_size=0.2
                )
                traditional_model.train(X_train, y_train)
            selected_model = traditional_model if traditional_model else transformer_model
        
        # Create voice analyzer for this session
        voice_analyzer = VoiceSentimentAnalyzer(selected_model)
        
        # Store session data with better initialization
        session_data = {
            'analyzer': voice_analyzer,
            'start_time': time.time(),
            'texts': [],
            'sentiments': [],
            'last_text': '',
            'status': 'starting'
        }
        active_voice_sessions[session_id] = session_data
        
        # Start real-time analysis with callbacks
        def on_text_callback(text):
            if session_id in active_voice_sessions:
                active_voice_sessions[session_id]['last_text'] = text
                active_voice_sessions[session_id]['texts'].append(text)
        
        def on_sentiment_callback(sentiment_data):
            if session_id in active_voice_sessions:
                active_voice_sessions[session_id]['sentiments'].append(sentiment_data)
        
        voice_analyzer.start_real_time_analysis(
            on_text_callback=on_text_callback,
            on_sentiment_callback=on_sentiment_callback
        )
        
        # Update session status
        active_voice_sessions[session_id]['status'] = 'recording'
        
        return jsonify({
            'status': 'success',
            'session_id': session_id,
            'message': 'Voice recording started'
        })
    
    except Exception as e:
        print(f"Error starting voice recording: {str(e)}")
        traceback.print_exc()
        return jsonify({
            'error': str(e),
            'status': 'error'
        }), 500

@app.route('/api/voice/stop_recording', methods=['POST'])
def stop_voice_recording():
    """Stop voice recording and get final results"""
    global active_voice_sessions
    
    try:
        data = request.get_json()
        session_id = data.get('session_id', 'default')
        
        if session_id not in active_voice_sessions:
            return jsonify({
                'error': 'No active session found',
                'status': 'error'
            }), 400
        
        session = active_voice_sessions[session_id]
        analyzer = session['analyzer']
        
        # Stop analysis and get summary
        summary = analyzer.stop_real_time_analysis()
        
        # Calculate session duration
        duration = time.time() - session['start_time']
        
        # Clean up session
        del active_voice_sessions[session_id]
        
        return jsonify({
            'status': 'success',
            'session_id': session_id,
            'duration': duration,
            'summary': summary
        })
    
    except Exception as e:
        print(f"Error stopping voice recording: {str(e)}")
        return jsonify({
            'error': str(e),
            'status': 'error'
        }), 500

@app.route('/api/voice/status', methods=['GET'])
def get_voice_status():
    """Get current voice recording status"""
    global active_voice_sessions
    
    session_id = request.args.get('session_id', 'default')
    
    if session_id in active_voice_sessions:
        session = active_voice_sessions[session_id]
        analyzer = session['analyzer']
        
        # Get the latest recognized text and sentiment
        latest_text = session.get('last_text', 'Listening...')
        latest_sentiment = None
        
        if session['sentiments']:
            latest_sentiment = session['sentiments'][-1]
        
        return jsonify({
            'status': 'recording',
            'session_id': session_id,
            'duration': time.time() - session['start_time'],
            'text_count': len(session['texts']),
            'sentiment_count': len(session['sentiments']),
            'latest_text': latest_text,
            'latest_sentiment': latest_sentiment,
            'recent_sentiments': session['sentiments'][-3:] if session['sentiments'] else []
        })
    else:
        return jsonify({
            'status': 'idle',
            'session_id': session_id
        })

@app.route('/api/voice/analyze_file', methods=['POST'])
def analyze_audio_file():
    """Analyze sentiment from uploaded audio file"""
    global transformer_model
    
    try:
        if 'audio' not in request.files:
            return jsonify({
                'error': 'No audio file provided',
                'status': 'error'
            }), 400
        
        file = request.files['audio']
        model_type = request.form.get('model_type', 'transformer')
        
        if file.filename == '':
            return jsonify({
                'error': 'No file selected',
                'status': 'error'
            }), 400
        
        # Check if models are loaded
        if transformer_model is None:
            return jsonify({
                'error': 'Models are still loading, please wait...',
                'status': 'loading'
            }), 503
        
        # Choose model
        if model_type == 'transformer':
            selected_model = transformer_model
        else:
            selected_model = traditional_model
        
        # Save uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as temp_file:
            file.save(temp_file.name)
            temp_file_path = temp_file.name
        
        try:
            # Create voice analyzer and process file
            voice_analyzer = VoiceSentimentAnalyzer(selected_model)
            result = voice_analyzer.analyze_audio_file(temp_file_path)
            
            return jsonify(result)
        
        finally:
            # Clean up temporary file
            if os.path.exists(temp_file_path):
                os.unlink(temp_file_path)
    
    except Exception as e:
        print(f"Error analyzing audio file: {str(e)}")
        return jsonify({
            'error': str(e),
            'status': 'error'
        }), 500

if __name__ == '__main__':
    print("Starting Sentiment Analysis GUI Server...")
    print("Models will load in the background...")
    print("Server will be available at: http://localhost:5000")
    
    app.run(debug=True, host='0.0.0.0', port=5000)
