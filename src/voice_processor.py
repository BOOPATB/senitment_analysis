

"""
Voice input processing module for sentiment analysis.
Handles speech-to-text conversion, real-time audio processing, and audio file handling.
"""

import os
import io
import time
import wave
import threading
import numpy as np
from typing import Optional, Callable, Dict, Any, List
import speech_recognition as sr
from pydub import AudioSegment
from pydub.effects import normalize
import pyaudio

class VoiceInputProcessor:
    """
    A comprehensive voice input processor for real-time speech recognition
    and audio processing for sentiment analysis.
    """
    
    def __init__(self, 
                 sample_rate: int = 16000,
                 chunk_duration: int = 30,  # milliseconds
                 vad_aggressiveness: int = 2):
        """
        Initialize the voice input processor.
        
        Args:
            sample_rate: Audio sample rate in Hz
            chunk_duration: Duration of each audio chunk in milliseconds
            vad_aggressiveness: Voice Activity Detection aggressiveness (0-3)
        """
        self.sample_rate = sample_rate
        self.chunk_duration = chunk_duration
        self.chunk_size = int(sample_rate * chunk_duration / 1000)
        self.vad_aggressiveness = vad_aggressiveness
        
        # Initialize components
        self.recognizer = sr.Recognizer()
        self.microphone = None
        
        # Recording state
        self.is_recording = False
        self.audio_data = []
        self.recording_thread = None
        
        # Callbacks
        self.on_speech_detected = None
        self.on_text_recognized = None
        self.on_audio_level_update = None
        
        # Audio settings
        self.energy_threshold = 300
        self.dynamic_energy_threshold = True
        self.pause_threshold = 0.8
        self.phrase_threshold = 0.3
        
        self._setup_microphone()
    
    def _setup_microphone(self):
        """Setup the microphone with optimal settings."""
        try:
            self.microphone = sr.Microphone(sample_rate=self.sample_rate)
            
            # Adjust for ambient noise
            print("Calibrating microphone for ambient noise...")
            with self.microphone as source:
                self.recognizer.adjust_for_ambient_noise(source, duration=1)
                
            # Set recognition parameters
            self.recognizer.energy_threshold = self.energy_threshold
            self.recognizer.dynamic_energy_threshold = self.dynamic_energy_threshold
            self.recognizer.pause_threshold = self.pause_threshold
            self.recognizer.phrase_threshold = self.phrase_threshold
            
            print(f"Microphone setup complete. Energy threshold: {self.recognizer.energy_threshold}")
            
        except Exception as e:
            print(f"Error setting up microphone: {e}")
            self.microphone = None
    
    def start_recording(self, 
                       on_text_callback: Optional[Callable[[str], None]] = None,
                       on_audio_level_callback: Optional[Callable[[float], None]] = None):
        """
        Start continuous voice recording and recognition.
        
        Args:
            on_text_callback: Callback function for recognized text
            on_audio_level_callback: Callback function for audio level updates
        """
        if self.is_recording:
            return
        
        if self.microphone is None:
            self._setup_microphone()
            if self.microphone is None:
                raise RuntimeError("Microphone not available")
        
        self.on_text_recognized = on_text_callback
        self.on_audio_level_update = on_audio_level_callback
        
        self.is_recording = True
        self.audio_data = []
        
        # Start recording in a separate thread
        self.recording_thread = threading.Thread(target=self._continuous_recording)
        self.recording_thread.daemon = True
        self.recording_thread.start()
    
    def stop_recording(self) -> Optional[str]:
        """
        Stop recording and return the final transcribed text.
        
        Returns:
            Final transcribed text or None if no speech detected
        """
        if not self.is_recording:
            return None
        
        self.is_recording = False
        
        # Wait for recording thread to finish
        if self.recording_thread and self.recording_thread.is_alive():
            self.recording_thread.join(timeout=2)
        
        # Process any remaining audio data
        if self.audio_data:
            return self._process_audio_data()
        
        return None
    
    def _continuous_recording(self):
        """Continuous recording loop with real-time processing."""
        try:
            with self.microphone as source:
                print("Starting continuous recording...")
                
                while self.is_recording:
                    try:
                        # Listen for audio with timeout
                        audio = self.recognizer.listen(source, timeout=0.5, phrase_time_limit=5)
                        
                        if audio:
                            # Calculate audio level
                            audio_level = self._calculate_audio_level(audio)
                            
                            if self.on_audio_level_update:
                                self.on_audio_level_update(audio_level)
                            
                            # Store audio data
                            self.audio_data.append(audio)
                            
                            # Process speech in background thread
                            threading.Thread(
                                target=self._process_speech_chunk,
                                args=(audio,),
                                daemon=True
                            ).start()
                    
                    except sr.WaitTimeoutError:
                        # No audio detected, continue listening
                        continue
                    except Exception as e:
                        print(f"Error in continuous recording: {e}")
                        time.sleep(0.1)
                        continue
        
        except Exception as e:
            print(f"Error in recording thread: {e}")
        finally:
            print("Recording stopped")
    
    def _process_speech_chunk(self, audio_data):
        """Process a chunk of audio data for speech recognition."""
        try:
            # Use Google Speech Recognition
            text = self.recognizer.recognize_google(audio_data, language='en-US')
            
            if text.strip() and self.on_text_recognized:
                self.on_text_recognized(text.strip())
                
        except sr.UnknownValueError:
            # Speech not understood
            pass
        except sr.RequestError as e:
            print(f"Could not request results from speech recognition service: {e}")
        except Exception as e:
            print(f"Error processing speech chunk: {e}")
    
    def _calculate_audio_level(self, audio_data) -> float:
        """Calculate the audio level (0.0 to 1.0) from audio data."""
        try:
            # Convert to numpy array
            audio_bytes = audio_data.get_raw_data()
            audio_array = np.frombuffer(audio_bytes, dtype=np.int16)
            
            # Calculate RMS level
            rms = np.sqrt(np.mean(audio_array**2))
            
            # Normalize to 0-1 range (approximate)
            normalized_level = min(rms / 3000, 1.0)
            
            return normalized_level
        
        except Exception as e:
            print(f"Error calculating audio level: {e}")
            return 0.0
    
    def _process_audio_data(self) -> Optional[str]:
        """Process accumulated audio data and return transcribed text."""
        if not self.audio_data:
            return None
        
        try:
            # Combine all audio chunks
            combined_audio = None
            for audio in self.audio_data:
                if combined_audio is None:
                    combined_audio = audio
                else:
                    # Combine audio data
                    combined_data = combined_audio.get_raw_data() + audio.get_raw_data()
                    combined_audio = sr.AudioData(
                        combined_data,
                        audio.sample_rate,
                        audio.sample_width
                    )
            
            if combined_audio:
                # Recognize the combined audio
                text = self.recognizer.recognize_google(combined_audio, language='en-US')
                return text.strip() if text.strip() else None
        
        except sr.UnknownValueError:
            return None
        except sr.RequestError as e:
            print(f"Could not request results from speech recognition service: {e}")
            return None
        except Exception as e:
            print(f"Error processing final audio data: {e}")
            return None
    
    def process_audio_file(self, file_path: str) -> Optional[str]:
        """
        Process an audio file and return transcribed text.
        
        Args:
            file_path: Path to the audio file
            
        Returns:
            Transcribed text or None if recognition failed
        """
        try:
            # Load audio file
            audio = AudioSegment.from_file(file_path)
            
            # Convert to required format (16kHz, mono, WAV)
            audio = audio.set_frame_rate(self.sample_rate)
            audio = audio.set_channels(1)
            
            # Normalize audio
            audio = normalize(audio)
            
            # Export to temporary WAV file
            temp_wav = io.BytesIO()
            audio.export(temp_wav, format="wav")
            temp_wav.seek(0)
            
            # Use speech recognition
            with sr.AudioFile(temp_wav) as source:
                audio_data = self.recognizer.record(source)
                text = self.recognizer.recognize_google(audio_data, language='en-US')
                return text.strip() if text.strip() else None
        
        except Exception as e:
            print(f"Error processing audio file {file_path}: {e}")
            return None
    
    def set_recognition_settings(self, 
                               energy_threshold: Optional[int] = None,
                               pause_threshold: Optional[float] = None,
                               phrase_threshold: Optional[float] = None):
        """
        Update speech recognition settings.
        
        Args:
            energy_threshold: Energy level threshold for speech detection
            pause_threshold: Seconds of non-speaking audio before a phrase is complete
            phrase_threshold: Minimum seconds of speaking audio before considering phrase
        """
        if energy_threshold is not None:
            self.recognizer.energy_threshold = energy_threshold
            self.energy_threshold = energy_threshold
            
        if pause_threshold is not None:
            self.recognizer.pause_threshold = pause_threshold
            self.pause_threshold = pause_threshold
            
        if phrase_threshold is not None:
            self.recognizer.phrase_threshold = phrase_threshold
            self.phrase_threshold = phrase_threshold
    
    def get_audio_devices(self) -> List[Dict[str, Any]]:
        """
        Get list of available audio input devices.
        
        Returns:
            List of device information dictionaries
        """
        try:
            devices = []
            for index, name in enumerate(sr.Microphone.list_microphone_names()):
                devices.append({
                    'index': index,
                    'name': name
                })
            return devices
        except Exception as e:
            print(f"Error getting audio devices: {e}")
            return []
    
    def test_microphone(self, duration: int = 3) -> Dict[str, Any]:
        """
        Test microphone functionality.
        
        Args:
            duration: Test duration in seconds
            
        Returns:
            Test results dictionary
        """
        if self.microphone is None:
            return {
                'success': False,
                'error': 'Microphone not available'
            }
        
        try:
            print(f"Testing microphone for {duration} seconds...")
            
            with self.microphone as source:
                # Adjust for ambient noise
                self.recognizer.adjust_for_ambient_noise(source, duration=1)
                print("Say something...")
                
                # Record audio
                audio = self.recognizer.listen(source, timeout=duration)
                
                # Try to recognize
                try:
                    text = self.recognizer.recognize_google(audio, language='en-US')
                    return {
                        'success': True,
                        'text': text,
                        'energy_threshold': self.recognizer.energy_threshold
                    }
                except sr.UnknownValueError:
                    return {
                        'success': True,
                        'text': None,
                        'message': 'Audio recorded but speech not understood',
                        'energy_threshold': self.recognizer.energy_threshold
                    }
        
        except Exception as e:
            return {
                'success': False,
                'error': str(e)
            }


class VoiceSentimentAnalyzer:
    """
    Real-time voice sentiment analysis combining speech recognition with sentiment analysis.
    """
    
    def __init__(self, sentiment_analyzer):
        """
        Initialize voice sentiment analyzer.
        
        Args:
            sentiment_analyzer: Sentiment analysis model (Traditional or Transformer)
        """
        self.sentiment_analyzer = sentiment_analyzer
        self.voice_processor = VoiceInputProcessor()
        
        # Real-time analysis settings
        self.text_buffer = []
        self.sentiment_history = []
        self.max_history_length = 10
        
        # Callbacks
        self.on_sentiment_update = None
        self.on_text_update = None
    
    def start_real_time_analysis(self,
                                on_sentiment_callback: Optional[Callable] = None,
                                on_text_callback: Optional[Callable] = None):
        """
        Start real-time voice sentiment analysis.
        
        Args:
            on_sentiment_callback: Callback for sentiment updates
            on_text_callback: Callback for text updates
        """
        self.on_sentiment_update = on_sentiment_callback
        self.on_text_update = on_text_callback
        
        # Start voice processing with callbacks
        self.voice_processor.start_recording(
            on_text_callback=self._handle_recognized_text,
            on_audio_level_callback=self._handle_audio_level
        )
    
    def stop_real_time_analysis(self) -> Dict[str, Any]:
        """
        Stop real-time analysis and return summary.
        
        Returns:
            Analysis summary dictionary
        """
        final_text = self.voice_processor.stop_recording()
        
        # Calculate overall sentiment
        if self.sentiment_history:
            sentiments = [s['sentiment'] for s in self.sentiment_history]
            from collections import Counter
            sentiment_counts = Counter(sentiments)
            overall_sentiment = sentiment_counts.most_common(1)[0][0]
        else:
            overall_sentiment = None
        
        return {
            'final_text': final_text,
            'text_buffer': self.text_buffer.copy(),
            'sentiment_history': self.sentiment_history.copy(),
            'overall_sentiment': overall_sentiment,
            'total_phrases': len(self.sentiment_history)
        }
    
    def _handle_recognized_text(self, text: str):
        """Handle newly recognized text and perform sentiment analysis."""
        print(f"Recognized: {text}")
        
        # Add to text buffer
        self.text_buffer.append(text)
        
        # Perform sentiment analysis
        try:
            if hasattr(self.sentiment_analyzer, 'predict_with_scores'):
                # Transformer model
                predictions = self.sentiment_analyzer.predict_with_scores([text])
                sentiment_data = predictions[0][0]  # Top prediction
                sentiment = sentiment_data['label']
                confidence = sentiment_data['score']
            else:
                # Traditional model
                sentiment = self.sentiment_analyzer.predict([text])[0]
                probabilities = self.sentiment_analyzer.predict_proba([text])[0]
                confidence = max(probabilities)
            
            # Create sentiment record
            sentiment_record = {
                'text': text,
                'sentiment': sentiment,
                'confidence': confidence,
                'timestamp': time.time()
            }
            
            # Add to history
            self.sentiment_history.append(sentiment_record)
            
            # Keep history size manageable
            if len(self.sentiment_history) > self.max_history_length:
                self.sentiment_history.pop(0)
            
            # Notify callbacks
            if self.on_text_update:
                self.on_text_update(text)
            
            if self.on_sentiment_update:
                self.on_sentiment_update(sentiment_record)
        
        except Exception as e:
            print(f"Error analyzing sentiment: {e}")
    
    def _handle_audio_level(self, level: float):
        """Handle audio level updates."""
        # This could be used for visual feedback
        pass
    
    def analyze_audio_file(self, file_path: str) -> Dict[str, Any]:
        """
        Analyze sentiment from an audio file.
        
        Args:
            file_path: Path to audio file
            
        Returns:
            Analysis results dictionary
        """
        # Transcribe audio
        text = self.voice_processor.process_audio_file(file_path)
        
        if not text:
            return {
                'success': False,
                'error': 'Could not transcribe audio'
            }
        
        # Analyze sentiment
        try:
            if hasattr(self.sentiment_analyzer, 'predict_with_scores'):
                # Transformer model
                predictions = self.sentiment_analyzer.predict_with_scores([text])
                sentiment_data = predictions[0]
            else:
                # Traditional model
                sentiment = self.sentiment_analyzer.predict([text])[0]
                probabilities = self.sentiment_analyzer.predict_proba([text])[0]
                
                # Convert to similar format
                labels = list(self.sentiment_analyzer.label_mapping.keys())
                sentiment_data = [
                    {'label': labels[i], 'score': float(prob)}
                    for i, prob in enumerate(probabilities)
                ]
                sentiment_data.sort(key=lambda x: x['score'], reverse=True)
            
            return {
                'success': True,
                'text': text,
                'sentiment_data': sentiment_data,
                'primary_sentiment': sentiment_data[0]['label'],
                'confidence': sentiment_data[0]['score']
            }
        
        except Exception as e:
            return {
                'success': False,
                'error': f'Sentiment analysis failed: {str(e)}'
            }


if __name__ == "__main__":
    # Test the voice processor
    processor = VoiceInputProcessor()
    
    # Test microphone
    print("Testing microphone...")
    result = processor.test_microphone(duration=5)
    print(f"Test result: {result}")
    
    if result['success']:
        print("Microphone test successful!")
        if result.get('text'):
            print(f"Recognized text: {result['text']}")
        else:
            print("No speech was recognized, but audio was captured.")
    else:
        print(f"Microphone test failed: {result.get('error', 'Unknown error')}")
