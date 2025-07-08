"""
State classification model wrapper.
This module provides an interface for state classification,
currently using pattern matching with future ONNX model support.
"""

import os
import numpy as np
from typing import Optional, Tuple, Dict
from pathlib import Path
import logging
from PIL import Image

# Import the pattern matcher as fallback
from state_patterns import StatePatternMatcher

logger = logging.getLogger(__name__)

# US states mapping
STATE_CLASSES = [
    'AL', 'AK', 'AZ', 'AR', 'CA', 'CO', 'CT', 'DE', 'FL', 'GA',
    'HI', 'ID', 'IL', 'IN', 'IA', 'KS', 'KY', 'LA', 'ME', 'MD',
    'MA', 'MI', 'MN', 'MS', 'MO', 'MT', 'NE', 'NV', 'NH', 'NJ',
    'NM', 'NY', 'NC', 'ND', 'OH', 'OK', 'OR', 'PA', 'RI', 'SC',
    'SD', 'TN', 'TX', 'UT', 'VT', 'VA', 'WA', 'WV', 'WI', 'WY'
]


class StateClassifier:
    """
    State classification for license plates.
    Supports both ONNX models and pattern-based fallback.
    """
    
    def __init__(self, model_path: Optional[str] = None, use_pattern_fallback: bool = True):
        """
        Initialize the state classifier.
        
        Args:
            model_path: Path to ONNX model file (optional)
            use_pattern_fallback: Whether to use pattern matching as fallback
        """
        self.model = None
        self.session = None
        self.use_pattern_fallback = use_pattern_fallback
        self.pattern_matcher = StatePatternMatcher() if use_pattern_fallback else None
        
        if model_path and os.path.exists(model_path):
            self._load_onnx_model(model_path)
        else:
            logger.info("No ONNX model found, using pattern-based classification")
    
    def _load_onnx_model(self, model_path: str):
        """Load ONNX model for inference."""
        try:
            import onnxruntime as ort
            
            self.session = ort.InferenceSession(model_path)
            self.model = True
            
            # Get model metadata
            self.input_name = self.session.get_inputs()[0].name
            self.input_shape = self.session.get_inputs()[0].shape
            self.output_name = self.session.get_outputs()[0].name
            
            logger.info(f"Loaded ONNX model from {model_path}")
            logger.info(f"Input shape: {self.input_shape}")
            
        except Exception as e:
            logger.error(f"Failed to load ONNX model: {e}")
            self.model = None
            self.session = None
    
    def classify_from_image(self, image: np.ndarray) -> Tuple[Optional[str], float, Dict[str, float]]:
        """
        Classify state from license plate image.
        
        Args:
            image: License plate image as numpy array (BGR or RGB)
            
        Returns:
            Tuple of (state_code, confidence, all_probabilities)
        """
        if self.model and self.session:
            return self._classify_with_model(image)
        else:
            # For now, return empty result for image-based classification
            # In a real implementation, you might use OCR + pattern matching
            return None, 0.0, {}
    
    def classify_from_text(self, plate_text: str) -> Tuple[Optional[str], float]:
        """
        Classify state from OCR text using pattern matching.
        
        Args:
            plate_text: OCR-extracted license plate text
            
        Returns:
            Tuple of (state_code, confidence)
        """
        if self.pattern_matcher:
            return self.pattern_matcher.extract_state_from_text(plate_text)
        return None, 0.0
    
    def _classify_with_model(self, image: np.ndarray) -> Tuple[Optional[str], float, Dict[str, float]]:
        """Run inference with ONNX model."""
        try:
            # Preprocess image
            processed = self._preprocess_image(image)
            
            # Run inference
            outputs = self.session.run([self.output_name], {self.input_name: processed})
            probabilities = outputs[0][0]  # Assuming batch size 1
            
            # Apply softmax if needed
            if not (probabilities.sum() > 0.99 and probabilities.sum() < 1.01):
                probabilities = self._softmax(probabilities)
            
            # Get top prediction
            top_idx = np.argmax(probabilities)
            confidence = float(probabilities[top_idx])
            state_code = STATE_CLASSES[top_idx] if top_idx < len(STATE_CLASSES) else None
            
            # Create probability dictionary
            prob_dict = {STATE_CLASSES[i]: float(probabilities[i]) 
                        for i in range(min(len(probabilities), len(STATE_CLASSES)))}
            
            return state_code, confidence, prob_dict
            
        except Exception as e:
            logger.error(f"Model inference failed: {e}")
            return None, 0.0, {}
    
    def _preprocess_image(self, image: np.ndarray) -> np.ndarray:
        """Preprocess image for model input."""
        # Expected input shape: (1, 3, 224, 224) for ResNet
        target_size = (self.input_shape[3], self.input_shape[2])  # (width, height)
        
        # Convert to PIL Image
        if len(image.shape) == 2:  # Grayscale
            pil_image = Image.fromarray(image)
            pil_image = pil_image.convert('RGB')
        else:  # Color
            pil_image = Image.fromarray(image)
        
        # Resize
        pil_image = pil_image.resize(target_size, Image.Resampling.LANCZOS)
        
        # Convert to numpy and normalize
        img_array = np.array(pil_image).astype(np.float32)
        
        # Normalize (ImageNet standards)
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        img_array = (img_array / 255.0 - mean) / std
        
        # Transpose to CHW format
        img_array = img_array.transpose(2, 0, 1)
        
        # Add batch dimension
        img_array = np.expand_dims(img_array, axis=0)
        
        return img_array
    
    def _softmax(self, x: np.ndarray) -> np.ndarray:
        """Apply softmax to logits."""
        exp_x = np.exp(x - np.max(x))
        return exp_x / exp_x.sum()
    
    def get_supported_states(self) -> list:
        """Get list of supported state codes."""
        if self.model:
            return STATE_CLASSES
        elif self.pattern_matcher:
            return self.pattern_matcher.get_supported_states()
        return []


# Global instance for easy access
_classifier = None

def get_state_classifier(model_path: Optional[str] = None) -> StateClassifier:
    """Get or create global state classifier instance."""
    global _classifier
    if _classifier is None:
        # Default model path
        if model_path is None:
            model_path = os.path.join(os.path.dirname(__file__), "..", "weights", "state_cls.onnx")
        _classifier = StateClassifier(model_path=model_path)
    return _classifier
