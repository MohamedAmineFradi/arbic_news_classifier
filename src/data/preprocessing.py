"""Prétraitement des textes arabes"""

import re
import string
from typing import List, Set
import nltk
from nltk.corpus import stopwords
from camel_tools.utils.normalize import normalize_unicode, normalize_alef_maksura_ar
from camel_tools.utils.dediac import dediac_ar
from camel_tools.tokenizers.word import simple_word_tokenize
import pyarabic.araby as araby
from config import PREPROCESSING_CONFIG, CUSTOM_STOPWORDS


class ArabicTextPreprocessor:
    """Préprocesseur de texte arabe"""
    
    def __init__(self, config=None):
        self.config = config or PREPROCESSING_CONFIG
        
        try:
            nltk.download('stopwords', quiet=True)
            self.stopwords = set(stopwords.words('arabic'))
        except:
            self.stopwords = set()
        
        self.stopwords.update(CUSTOM_STOPWORDS)
        self.arabic_punctuation = '،؛؟!«»٪×÷'
        self.english_punctuation = string.punctuation
        self.all_punctuation = self.arabic_punctuation + self.english_punctuation
    
    def remove_diacritics(self, text: str) -> str:
        """
        Remove diacritics (tashkeel)
        """
        return dediac_ar(text)
    
    def normalize_text(self, text: str) -> str:
        """
        Normalize Arabic text
        """
        #  Unicode
        text = normalize_unicode(text)
        
        text = re.sub('[إأآا]', 'ا', text)
        
        text = re.sub('ة', 'ه', text)
        
        text = normalize_alef_maksura_ar(text)
        
        return text
    
    def remove_punctuation(self, text: str) -> str:
        """
        Remove punctuation
        """
        translator = str.maketrans('', '', self.all_punctuation)
        return text.translate(translator)
    
    def remove_english(self, text: str) -> str:
        """
        Remove English letters
        """
        return re.sub(r'[a-zA-Z]+', '', text)
    
    def remove_numbers(self, text: str) -> str:
        """
        Remove numbers
        """
        text = re.sub(r'[0-9]+', '', text)
        text = re.sub(r'[٠-٩]+', '', text)
        return text
    
    def remove_urls(self, text: str) -> str:
        """
        Remove URLs
        """
        url_pattern = r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
        return re.sub(url_pattern, '', text)
    
    def remove_mentions(self, text: str) -> str:
        """
        Remove mentions
        """
        return re.sub(r'@[\w]+', '', text)
    
    def remove_hashtags(self, text: str) -> str:
        """
        Remove hashtags
        """
        return re.sub(r'#[\w]+', '', text)
    
    def remove_extra_whitespace(self, text: str) -> str:
        """
        Remove extra whitespace
        """
        return ' '.join(text.split())
    
    def tokenize(self, text: str) -> List[str]:
        """
        Tokenize text
        """
        return simple_word_tokenize(text)
    
    def remove_stopwords(self, tokens: List[str]) -> List[str]:
        """
        Remove stopwords
        """
        return [token for token in tokens if token not in self.stopwords]
    
    def stem_text(self, tokens: List[str]) -> List[str]:
        """
        Simple stemming
        """
        #  pyarabic / stemming 
        stemmed = []
        for token in tokens:
            stem = araby.strip_tashkeel(token)
            stem = araby.strip_tatweel(stem)
            stemmed.append(stem)
        return stemmed
    
    def preprocess(self, text: str) -> str:
        """Traitement complet du texte"""
        if not isinstance(text, str):
            return ""
        
        text = self.remove_urls(text)
        text = self.remove_mentions(text)
        text = self.remove_hashtags(text)
        
        if self.config.get('remove_diacritics', True):
            text = self.remove_diacritics(text)
        
        if self.config.get('normalize_arabic', True):
            text = self.normalize_text(text)
        
        if self.config.get('remove_punctuation', True):
            text = self.remove_punctuation(text)
        
        if self.config.get('remove_english', True):
            text = self.remove_english(text)
        
        if self.config.get('remove_numbers', False):
            text = self.remove_numbers(text)
        
        text = self.remove_extra_whitespace(text)
        tokens = self.tokenize(text)
        
        if self.config.get('remove_stopwords', True):
            tokens = self.remove_stopwords(tokens)
        
        if self.config.get('apply_stemming', True):
            tokens = self.stem_text(tokens)
        
        tokens = [t for t in tokens if len(t) > 1]
        return ' '.join(tokens)
    
    def preprocess_batch(self, texts: List[str]) -> List[str]:
        """Traiter un lot de textes"""
        from tqdm import tqdm
        return [self.preprocess(text) for text in tqdm(texts, desc="Traitement")]
