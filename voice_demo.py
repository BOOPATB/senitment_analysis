#!/usr/bin/env python3
"""
Voice Sentiment Analysis Demo
A standalone script to test the voice input functionality with sentiment analysis.
"""

import sys
import os
import time
sys.path.append('src')

from voice_processor import VoiceInputProcessor, VoiceSentimentAnalyzer
from sentiment_analyzer import TransformerSentimentAnalyzer, TraditionalSentimentAnalyzer
from preprocessing import load_sample_data, create_train_test_split, TextPreprocessor


def test_voice_processor():
    """Test the basic voice processing functionality."""
    print("=== Testing Voice Processor ===")
    
    try:
        processor = VoiceInputProcessor()
        
        # Test microphone
        print("\n1. Testing microphone...")
        result = processor.test_microphone(duration=5)
        
        if result['success']:
            print("✅ Microphone test successful!")
            if result.get('text'):
                print(f"   Recognized text: '{result['text']}'")
            else:
                print("   Audio captured but no speech recognized.")
        else:
            print(f"❌ Microphone test failed: {result.get('error', 'Unknown error')}")
            return False
        
        # List audio devices
        print("\n2. Available audio devices:")
        devices = processor.get_audio_devices()
        for device in devices:
            print(f"   {device['index']}: {device['name']}")
        
        return True
    
    except Exception as e:
        print(f"❌ Error testing voice processor: {e}")
        return False


def test_voice_sentiment_analysis():
    """Test the voice sentiment analysis functionality."""
    print("\n=== Testing Voice Sentiment Analysis ===")
    
    try:
        # Load and prepare models
        print("\n1. Loading sentiment analysis models...")
        
        # Load transformer model
        print("   Loading transformer model...")
        transformer_model = TransformerSentimentAnalyzer()
        print("   ✅ Transformer model loaded")
        
        # Load and train traditional model
        print("   Loading traditional model...")
        traditional_model = TraditionalSentimentAnalyzer()
        
        # Quick training
        print("   Training traditional model with sample data...")
        preprocessor = TextPreprocessor()
        df = load_sample_data()
        df_processed = preprocessor.preprocess_dataframe(df, 'text', 'sentiment')
        X_train, _, y_train, _ = create_train_test_split(
            df_processed, 'text_processed', 'sentiment', test_size=0.2
        )
        traditional_model.train(X_train, y_train)
        print("   ✅ Traditional model trained")
        
        # Create voice sentiment analyzer
        print("\n2. Creating voice sentiment analyzer...")
        voice_analyzer = VoiceSentimentAnalyzer(transformer_model)
        print("   ✅ Voice analyzer created")
        
        # Test real-time analysis
        print("\n3. Starting real-time voice sentiment analysis...")
        print("   🎤 Speak now! Say something positive, negative, or neutral.")
        print("   🛑 Press Ctrl+C to stop recording\n")
        
        # Set up callbacks for real-time feedback
        def on_text_callback(text):
            print(f"   🗣️  Heard: \"{text}\"")
        
        def on_sentiment_callback(sentiment_data):
            sentiment = sentiment_data['sentiment']
            confidence = sentiment_data['confidence']
            
            # Add color coding
            color = {
                'positive': '\033[92m',  # Green
                'negative': '\033[91m',  # Red
                'neutral': '\033[93m'    # Yellow
            }.get(sentiment.lower(), '')
            reset = '\033[0m'
            
            print(f"   💭 Sentiment: {color}{sentiment.upper()}{reset} "
                  f"(confidence: {confidence:.2f})")
        
        # Start analysis
        voice_analyzer.start_real_time_analysis(
            on_sentiment_callback=on_sentiment_callback,
            on_text_callback=on_text_callback
        )
        
        try:
            # Keep running until interrupted
            while True:
                time.sleep(1)
        
        except KeyboardInterrupt:
            print("\n\n⏹️  Stopping recording...")
        
        # Stop analysis and get summary
        summary = voice_analyzer.stop_real_time_analysis()
        
        # Display results
        print("\n=== Analysis Summary ===")
        print(f"Total phrases detected: {summary['total_phrases']}")
        print(f"Overall sentiment: {summary['overall_sentiment']}")
        print(f"Final transcribed text: {summary.get('final_text', 'None')}")
        
        if summary['sentiment_history']:
            print("\nDetailed sentiment history:")
            for i, entry in enumerate(summary['sentiment_history'], 1):
                timestamp = time.strftime('%H:%M:%S', time.localtime(entry['timestamp']))
                print(f"  {i}. [{timestamp}] \"{entry['text']}\" → "
                      f"{entry['sentiment']} ({entry['confidence']:.2f})")
        
        return True
    
    except Exception as e:
        print(f"❌ Error in voice sentiment analysis: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_audio_file_analysis():
    """Test audio file analysis if files are available."""
    print("\n=== Testing Audio File Analysis ===")
    
    try:
        # Check for sample audio files
        audio_dir = "sample_audio"
        if not os.path.exists(audio_dir):
            print("ℹ️  No sample audio directory found. Skipping file analysis test.")
            print("   You can create a 'sample_audio' folder and add .wav files to test this feature.")
            return True
        
        audio_files = [f for f in os.listdir(audio_dir) if f.endswith(('.wav', '.mp3', '.m4a'))]
        if not audio_files:
            print("ℹ️  No audio files found in sample_audio directory.")
            return True
        
        # Load model
        transformer_model = TransformerSentimentAnalyzer()
        voice_analyzer = VoiceSentimentAnalyzer(transformer_model)
        
        print(f"Found {len(audio_files)} audio files:")
        
        for audio_file in audio_files[:3]:  # Test first 3 files
            file_path = os.path.join(audio_dir, audio_file)
            print(f"\n  Analyzing: {audio_file}")
            
            result = voice_analyzer.analyze_audio_file(file_path)
            
            if result['success']:
                print(f"    Text: \"{result['text']}\"")
                print(f"    Sentiment: {result['primary_sentiment']} "
                      f"({result['confidence']:.2f})")
            else:
                print(f"    ❌ Error: {result['error']}")
        
        return True
    
    except Exception as e:
        print(f"❌ Error in audio file analysis: {e}")
        return False


def main():
    """Main demo function."""
    print("🎙️ Voice Sentiment Analysis Demo")
    print("=" * 50)
    
    print("\nThis demo will test the voice input functionality for sentiment analysis.")
    print("Make sure you have:")
    print("1. A working microphone")
    print("2. Internet connection (for Google Speech Recognition)")
    print("3. The required dependencies installed")
    
    input("\nPress Enter to continue...")
    
    # Test 1: Basic voice processing
    if not test_voice_processor():
        print("\n❌ Voice processor test failed. Please check your microphone setup.")
        return
    
    # Test 2: Voice sentiment analysis
    if not test_voice_sentiment_analysis():
        print("\n❌ Voice sentiment analysis test failed.")
        return
    
    # Test 3: Audio file analysis (optional)
    test_audio_file_analysis()
    
    print("\n✅ Demo completed successfully!")
    print("\nNext steps:")
    print("1. Run 'python launch_gui.py' to start the web interface")
    print("2. Use the voice input features in the GUI")
    print("3. Try uploading audio files for batch analysis")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n👋 Demo interrupted by user.")
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
