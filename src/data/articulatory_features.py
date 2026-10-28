"""
Articulatory feature mappings for phoneme classification.

This module provides mappings from phonemes to articulatory features:
- Manner of articulation: vowel, plosive, fricative, nasal, etc.
- Place of articulation: labial, alveolar, velar, etc.
- Voicing: voiced, unvoiced
"""

from typing import Dict, List, Optional
import numpy as np


# Phoneme to Manner of Articulation
PHONEME_TO_MANNER = {
    # Vowels
    'iy': 'vowel', 'ih': 'vowel', 'eh': 'vowel', 'ae': 'vowel', 'ah': 'vowel',
    'uw': 'vowel', 'uh': 'vowel', 'aa': 'vowel', 'ey': 'vowel', 'ay': 'vowel',
    'oy': 'vowel', 'aw': 'vowel', 'ow': 'vowel', 'er': 'vowel',
    
    # Plosives (stops)
    'p': 'plosive', 'b': 'plosive', 't': 'plosive', 
    'd': 'plosive', 'k': 'plosive', 'g': 'plosive',
    
    # Fricatives
    'f': 'fricative', 'v': 'fricative', 's': 'fricative', 
    'z': 'fricative', 'sh': 'fricative', 'zh': 'fricative',
    'th': 'fricative', 'dh': 'fricative', 'h': 'fricative',
    
    # Nasals
    'm': 'nasal', 'n': 'nasal', 'ng': 'nasal',
    
    # Liquids
    'l': 'liquid', 'r': 'liquid',
    
    # Glides (semivowels)
    'w': 'glide', 'y': 'glide',
    
    # Affricates
    'ch': 'affricate', 'jh': 'affricate',
}

# Phoneme to Place of Articulation (for consonants)
PHONEME_TO_PLACE = {
    # Labial (lips)
    'p': 'labial', 'b': 'labial', 'm': 'labial', 
    'f': 'labial', 'v': 'labial', 'w': 'labial',
    
    # Alveolar (tongue tip to alveolar ridge)
    't': 'alveolar', 'd': 'alveolar', 'n': 'alveolar', 
    's': 'alveolar', 'z': 'alveolar', 'l': 'alveolar',
    
    # Velar (back of tongue to soft palate)
    'k': 'velar', 'g': 'velar', 'ng': 'velar',
    
    # Palatal (tongue to hard palate)
    'sh': 'palatal', 'zh': 'palatal', 'ch': 'palatal', 
    'jh': 'palatal', 'y': 'palatal', 'r': 'palatal',
    
    # Dental (tongue to teeth)
    'th': 'dental', 'dh': 'dental',
    
    # Glottal
    'h': 'glottal',
    
    # Vowels don't have place (or could be 'vowel')
    'iy': 'vowel', 'ih': 'vowel', 'eh': 'vowel', 'ae': 'vowel', 'ah': 'vowel',
    'uw': 'vowel', 'uh': 'vowel', 'aa': 'vowel', 'ey': 'vowel', 'ay': 'vowel',
    'oy': 'vowel', 'aw': 'vowel', 'ow': 'vowel', 'er': 'vowel',
}

# Phoneme to Voicing
PHONEME_TO_VOICING = {
    # Unvoiced consonants
    'p': 'unvoiced', 't': 'unvoiced', 'k': 'unvoiced',
    'f': 'unvoiced', 's': 'unvoiced', 'sh': 'unvoiced', 
    'th': 'unvoiced', 'h': 'unvoiced', 'ch': 'unvoiced',
    
    # Voiced consonants
    'b': 'voiced', 'd': 'voiced', 'g': 'voiced',
    'v': 'voiced', 'z': 'voiced', 'zh': 'voiced', 
    'dh': 'voiced', 'jh': 'voiced',
    'm': 'voiced', 'n': 'voiced', 'ng': 'voiced',
    'l': 'voiced', 'r': 'voiced', 'w': 'voiced', 'y': 'voiced',
    
    # All vowels are voiced
    'iy': 'voiced', 'ih': 'voiced', 'eh': 'voiced', 'ae': 'voiced', 'ah': 'voiced',
    'uw': 'voiced', 'uh': 'voiced', 'aa': 'voiced', 'ey': 'voiced', 'ay': 'voiced',
    'oy': 'voiced', 'aw': 'voiced', 'ow': 'voiced', 'er': 'voiced',
}

# Simplified binary classifications
PHONEME_TO_CONSONANT_VOWEL = {
    **{p: 'consonant' for p in ['p', 'b', 't', 'd', 'k', 'g', 'f', 'v', 's', 'z', 
                                  'sh', 'zh', 'th', 'dh', 'h', 'm', 'n', 'ng', 
                                  'l', 'r', 'w', 'y', 'ch', 'jh']},
    **{p: 'vowel' for p in ['iy', 'ih', 'eh', 'ae', 'ah', 'uw', 'uh', 'aa', 
                             'ey', 'ay', 'oy', 'aw', 'ow', 'er']},
}

# Articulatory groups for TMS analysis
TONGUE_PHONEMES = ['t', 'd', 'n', 's', 'z', 'l', 'r', 'th', 'dh']  # Require tongue movement
LIP_PHONEMES = ['p', 'b', 'm', 'f', 'v', 'w']  # Require lip movement


class ArticulatoryFeatureEncoder:
    """
    Encoder for converting phoneme labels to articulatory features.
    """
    
    def __init__(self, feature_type: str = 'manner'):
        """
        Initialize encoder.
        
        Args:
            feature_type: Type of feature ('manner', 'place', 'voicing', 'consonant_vowel')
        """
        self.feature_type = feature_type
        
        if feature_type == 'manner':
            self.mapping = PHONEME_TO_MANNER
        elif feature_type == 'place':
            self.mapping = PHONEME_TO_PLACE
        elif feature_type == 'voicing':
            self.mapping = PHONEME_TO_VOICING
        elif feature_type == 'consonant_vowel':
            self.mapping = PHONEME_TO_CONSONANT_VOWEL
        else:
            raise ValueError(f"Unknown feature type: {feature_type}")
        
        # Create label mappings
        self.unique_labels = sorted(set(self.mapping.values()))
        self.label_to_idx = {label: idx for idx, label in enumerate(self.unique_labels)}
        self.idx_to_label = {idx: label for label, idx in self.label_to_idx.items()}
        
    @property
    def n_classes(self) -> int:
        """Number of classes for this feature."""
        return len(self.unique_labels)
    
    def encode_phoneme(self, phoneme: str) -> int:
        """
        Convert phoneme to feature class index.
        
        Args:
            phoneme: Phoneme string (e.g., 'p', 'iy')
            
        Returns:
            Class index for the feature
        """
        phoneme = phoneme.lower()
        if phoneme not in self.mapping:
            raise ValueError(f"Unknown phoneme: {phoneme}")
        
        feature_label = self.mapping[phoneme]
        return self.label_to_idx[feature_label]
    
    def encode_phonemes(self, phonemes: List[str]) -> np.ndarray:
        """
        Convert list of phonemes to feature class indices.
        
        Args:
            phonemes: List of phoneme strings
            
        Returns:
            Array of class indices
        """
        return np.array([self.encode_phoneme(p) for p in phonemes])
    
    def decode_index(self, idx: int) -> str:
        """
        Convert feature class index back to feature label.
        
        Args:
            idx: Class index
            
        Returns:
            Feature label string
        """
        return self.idx_to_label[idx]
    
    def decode_indices(self, indices: np.ndarray) -> List[str]:
        """
        Convert array of indices to feature labels.
        
        Args:
            indices: Array of class indices
            
        Returns:
            List of feature label strings
        """
        return [self.decode_index(int(idx)) for idx in indices]
    
    def get_class_names(self) -> List[str]:
        """Get list of class names in order."""
        return self.unique_labels
    
    def get_phoneme_mapping(self) -> Dict[str, str]:
        """Get the phoneme to feature mapping."""
        return self.mapping.copy()


def get_feature_distribution(phonemes: List[str], feature_type: str = 'manner') -> Dict[str, int]:
    """
    Get distribution of features in a phoneme list.
    
    Args:
        phonemes: List of phonemes
        feature_type: Type of feature to analyze
        
    Returns:
        Dictionary mapping feature labels to counts
    """
    encoder = ArticulatoryFeatureEncoder(feature_type)
    features = encoder.encode_phonemes(phonemes)
    feature_labels = encoder.decode_indices(features)
    
    distribution = {}
    for label in feature_labels:
        distribution[label] = distribution.get(label, 0) + 1
    
    return distribution


def analyze_tms_phonemes() -> Dict[str, List[str]]:
    """
    Get phoneme groups for TMS analysis.
    
    Returns:
        Dictionary with 'tongue' and 'lip' phoneme lists
    """
    return {
        'tongue': TONGUE_PHONEMES.copy(),
        'lip': LIP_PHONEMES.copy(),
    }


# Example usage and testing
if __name__ == '__main__':
    print("Articulatory Feature Encoder Examples\n")
    
    # Test manner encoder
    manner_encoder = ArticulatoryFeatureEncoder('manner')
    print(f"Manner classes ({manner_encoder.n_classes}): {manner_encoder.get_class_names()}")
    print(f"  'p' -> {manner_encoder.encode_phoneme('p')} ({manner_encoder.decode_index(manner_encoder.encode_phoneme('p'))})")
    print(f"  'iy' -> {manner_encoder.encode_phoneme('iy')} ({manner_encoder.decode_index(manner_encoder.encode_phoneme('iy'))})")
    print()
    
    # Test place encoder
    place_encoder = ArticulatoryFeatureEncoder('place')
    print(f"Place classes ({place_encoder.n_classes}): {place_encoder.get_class_names()}")
    print(f"  'p' -> {place_encoder.encode_phoneme('p')} ({place_encoder.decode_index(place_encoder.encode_phoneme('p'))})")
    print(f"  't' -> {place_encoder.encode_phoneme('t')} ({place_encoder.decode_index(place_encoder.encode_phoneme('t'))})")
    print()
    
    # Test voicing encoder
    voicing_encoder = ArticulatoryFeatureEncoder('voicing')
    print(f"Voicing classes ({voicing_encoder.n_classes}): {voicing_encoder.get_class_names()}")
    print(f"  'p' -> {voicing_encoder.encode_phoneme('p')} ({voicing_encoder.decode_index(voicing_encoder.encode_phoneme('p'))})")
    print(f"  'b' -> {voicing_encoder.encode_phoneme('b')} ({voicing_encoder.decode_index(voicing_encoder.encode_phoneme('b'))})")
    print()
    
    # Test consonant/vowel encoder
    cv_encoder = ArticulatoryFeatureEncoder('consonant_vowel')
    print(f"Consonant/Vowel classes ({cv_encoder.n_classes}): {cv_encoder.get_class_names()}")
    print(f"  'p' -> {cv_encoder.encode_phoneme('p')} ({cv_encoder.decode_index(cv_encoder.encode_phoneme('p'))})")
    print(f"  'iy' -> {cv_encoder.encode_phoneme('iy')} ({cv_encoder.decode_index(cv_encoder.encode_phoneme('iy'))})")
    print()
    
    # Test distribution
    test_phonemes = ['p', 'b', 't', 'd', 'iy', 'ih', 'eh']
    print(f"Feature distribution for {test_phonemes}:")
    print(f"  Manner: {get_feature_distribution(test_phonemes, 'manner')}")
    print(f"  Place: {get_feature_distribution(test_phonemes, 'place')}")
    print(f"  Voicing: {get_feature_distribution(test_phonemes, 'voicing')}")

