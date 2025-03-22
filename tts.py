import os
from gtts import gTTS
import tempfile
from typing import Dict, Any, Optional
import re
from deep_translator import GoogleTranslator  # Using deep-translator for full translation

class TextToSpeechConverter:
    """Class for converting text to Hindi speech."""
    
    def __init__(self):
        self.temp_dir = tempfile.gettempdir()
    
    def clean_text_for_tts(self, text: str) -> str:
        """Clean and prepare text for TTS conversion."""
        if not text:
            return "No text provided for conversion."
            
        text = re.sub(r'http\S+', '', text)  # Remove URLs
        text = re.sub(r'[^\w\s.,?!;:\-\'"()]', ' ', text)  # Remove special characters
        text = re.sub(r'\s+', ' ', text).strip()  # Remove extra whitespace
        return text
    
    def translate_to_hindi(self, text: str) -> str:
        """Translate full text to Hindi using Google Translator."""
        if not text:
            return "अनुवाद के लिए कोई पाठ प्रदान नहीं किया गया।"
            
        try:
            # Limit text length to avoid API issues
            text = text[:500]  # Limiting to 500 chars for stability
            hindi_text = GoogleTranslator(source='auto', target='hi').translate(text)
            if not hindi_text:
                return "अनुवाद उपलब्ध नहीं है।"  # Translation not available
            return hindi_text
        except Exception as e:
            print(f"Translation error: {str(e)}")
            return "अनुवाद में त्रुटि हुई। कृपया पुनः प्रयास करें।"  # Translation error fallback
    
    def generate_speech(self, text: str, output_file: Optional[str] = None) -> Dict[str, Any]:
        """Generate Hindi speech from full translated text."""
        if not text:
            return {
                "success": False,
                "message": "No text provided for speech generation",
                "output_file": None,
                "hindi_text": None
            }
            
        try:
            cleaned_text = self.clean_text_for_tts(text)
            hindi_text = self.translate_to_hindi(cleaned_text)
            
            if not output_file:
                output_file = os.path.join(self.temp_dir, f"news_summary_{abs(hash(hindi_text) % 10000)}.mp3")
            
            tts = gTTS(text=hindi_text, lang='hi', slow=False)
            tts.save(output_file)
            
            return {
                "success": True,
                "message": "Speech generated successfully",
                "output_file": output_file,
                "hindi_text": hindi_text
            }
        except Exception as e:
            print(f"Speech generation error: {str(e)}")
            return {
                "success": False,
                "message": f"Error generating speech: {str(e)}",
                "output_file": None,
                "hindi_text": None
            }
    
    def generate_summary_speech(self, report: Dict[str, Any]) -> Dict[str, Any]:
        """Generate speech for a translated news summary."""
        if not report:
            return {
                "success": False,
                "message": "No report data provided",
                "output_file": None,
                "hindi_text": None
            }
            
        try:
            company = report.get("company", "Unknown company")
            final_sentiment = report.get("final_sentiment_analysis", "No sentiment analysis available")
            
            summary_text = f"News report for {company}. {final_sentiment}"
            
            if "sentiment_counts" in report:
                counts = report["sentiment_counts"]
                summary_text += f" Found {counts.get('positive', 0)} positive, {counts.get('negative', 0)} negative, and {counts.get('neutral', 0)} neutral articles."
            
            result = self.generate_speech(summary_text)
            
            return result
        except Exception as e:
            print(f"Summary speech error: {str(e)}")
            return {
                "success": False,
                "message": f"Error generating summary speech: {str(e)}",
                "output_file": None,
                "hindi_text": None
            }