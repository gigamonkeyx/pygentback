"""
Data Processing Utilities

This module provides utilities for data processing, validation, transformation,
and serialization including text processing, JSON handling, and schema validation.
"""

import json
import re
import hashlib
import base64
from typing import Any, Dict, List, Optional, Union, Callable, Type
from dataclasses import dataclass, asdict
from datetime import datetime, date
from pathlib import Path
import logging
from jsonschema import validate, ValidationError
import yaml

logger = logging.getLogger(__name__)


@dataclass
class ValidationResult:
    """Result of data validation"""
    is_valid: bool
    errors: List[str]
    warnings: List[str]
    
    def add_error(self, error: str):
        """Add validation error"""
        self.errors.append(error)
        self.is_valid = False
    
    def add_warning(self, warning: str):
        """Add validation warning"""
        self.warnings.append(warning)


class DataProcessor:
    """Generic data processor with transformation capabilities"""
    
    def __init__(self):
        self.transformers: Dict[str, Callable] = {}
        self.validators: Dict[str, Callable] = {}
    
    def add_transformer(self, name: str, transformer: Callable[[Any], Any]):
        """Add data transformer"""
        self.transformers[name] = transformer
    
    def add_validator(self, name: str, validator: Callable[[Any], ValidationResult]):
        """Add data validator"""
        self.validators[name] = validator
    
    def transform(self, data: Any, transformer_name: str) -> Any:
        """Transform data using named transformer"""
        if transformer_name not in self.transformers:
            raise ValueError(f"Unknown transformer: {transformer_name}")
        
        return self.transformers[transformer_name](data)
    
    def validate(self, data: Any, validator_name: str) -> ValidationResult:
        """Validate data using named validator"""
        if validator_name not in self.validators:
            raise ValueError(f"Unknown validator: {validator_name}")
        
        return self.validators[validator_name](data)
    
    def process(self, data: Any, 
                transformers: Optional[List[str]] = None,
                validators: Optional[List[str]] = None) -> Dict[str, Any]:
        """Process data with transformers and validators"""
        result = {
            'original_data': data,
            'transformed_data': data,
            'validation_results': {},
            'success': True,
            'errors': []
        }
        
        # Apply transformers
        if transformers:
            for transformer_name in transformers:
                try:
                    result['transformed_data'] = self.transform(
                        result['transformed_data'], 
                        transformer_name
                    )
                except Exception as e:
                    result['errors'].append(f"Transformer '{transformer_name}' failed: {str(e)}")
                    result['success'] = False
        
        # Apply validators
        if validators:
            for validator_name in validators:
                try:
                    validation_result = self.validate(
                        result['transformed_data'], 
                        validator_name
                    )
                    result['validation_results'][validator_name] = validation_result
                    
                    if not validation_result.is_valid:
                        result['success'] = False
                        result['errors'].extend(validation_result.errors)
                        
                except Exception as e:
                    result['errors'].append(f"Validator '{validator_name}' failed: {str(e)}")
                    result['success'] = False
        
        return result


class TextProcessor:
    """Text processing utilities"""
    
    @staticmethod
    def clean_text(text: str, 
                   remove_extra_whitespace: bool = True,
                   remove_special_chars: bool = False,
                   lowercase: bool = False) -> str:
        """Clean and normalize text"""
        if not isinstance(text, str):
            text = str(text)
        
        # Remove extra whitespace
        if remove_extra_whitespace:
            text = re.sub(r'\s+', ' ', text.strip())
        
        # Remove special characters
        if remove_special_chars:
            text = re.sub(r'[^\w\s]', '', text)
        
        # Convert to lowercase
        if lowercase:
            text = text.lower()
        
        return text
    
    @staticmethod
    def extract_emails(text: str) -> List[str]:
        """Extract email addresses from text"""
        email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
        return re.findall(email_pattern, text)
    
    @staticmethod
    def extract_urls(text: str) -> List[str]:
        """Extract URLs from text"""
        url_pattern = r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
        return re.findall(url_pattern, text)
    
    @staticmethod
    def extract_phone_numbers(text: str) -> List[str]:
        """Extract phone numbers from text"""
        phone_pattern = r'(\+?1[-.\s]?)?\(?([0-9]{3})\)?[-.\s]?([0-9]{3})[-.\s]?([0-9]{4})'
        matches = re.findall(phone_pattern, text)
        return [''.join(match) for match in matches]
    
    @staticmethod
    def tokenize(text: str, method: str = "simple") -> List[str]:
        """Tokenize text"""
        if method == "simple":
            return text.split()
        elif method == "words":
            return re.findall(r'\b\w+\b', text.lower())
        elif method == "sentences":
            return re.split(r'[.!?]+', text)
        else:
            raise ValueError(f"Unknown tokenization method: {method}")
    
    @staticmethod
    def calculate_similarity(text1: str, text2: str, method: str = "jaccard") -> float:
        """Calculate text similarity"""
        if method == "jaccard":
            set1 = set(TextProcessor.tokenize(text1, "words"))
            set2 = set(TextProcessor.tokenize(text2, "words"))
            
            intersection = len(set1.intersection(set2))
            union = len(set1.union(set2))
            
            return intersection / union if union > 0 else 0.0
        else:
            raise ValueError(f"Unknown similarity method: {method}")
    
    @staticmethod
    def generate_hash(text: str, algorithm: str = "sha256") -> str:
        """Generate hash of text"""
        if algorithm == "md5":
            return hashlib.md5(text.encode()).hexdigest()
        elif algorithm == "sha256":
            return hashlib.sha256(text.encode()).hexdigest()
        elif algorithm == "sha1":
            return hashlib.sha1(text.encode()).hexdigest()
        else:
            raise ValueError(f"Unknown hash algorithm: {algorithm}")


class JSONProcessor:
    """JSON processing utilities"""
    
    @staticmethod
    def safe_load(json_str: str, default: Any = None) -> Any:
        """Safely load JSON with fallback"""
        try:
            return json.loads(json_str)
        except (json.JSONDecodeError, TypeError) as e:
            logger.warning(f"Failed to parse JSON: {e}")
            return default
    
    @staticmethod
    def safe_dump(data: Any, default: Any = None, **kwargs) -> str:
        """Safely dump JSON with custom serializer"""
        def json_serializer(obj):
            if isinstance(obj, (datetime, date)):
                return obj.isoformat()
            elif hasattr(obj, '__dict__'):
                return obj.__dict__
            elif default is not None:
                return default
            else:
                return str(obj)
        
        try:
            return json.dumps(data, default=json_serializer, **kwargs)
        except Exception as e:
            logger.error(f"Failed to serialize JSON: {e}")
            return "{}"
    
    @staticmethod
    def flatten(data: Dict[str, Any], separator: str = ".") -> Dict[str, Any]:
        """Flatten nested JSON structure"""
        def _flatten(obj, parent_key=""):
            items = []
            if isinstance(obj, dict):
                for key, value in obj.items():
                    new_key = f"{parent_key}{separator}{key}" if parent_key else key
                    items.extend(_flatten(value, new_key).items())
            elif isinstance(obj, list):
                for i, value in enumerate(obj):
                    new_key = f"{parent_key}{separator}{i}" if parent_key else str(i)
                    items.extend(_flatten(value, new_key).items())
            else:
                return {parent_key: obj}
            return dict(items)
        
        return _flatten(data)
    
    @staticmethod
    def unflatten(data: Dict[str, Any], separator: str = ".") -> Dict[str, Any]:
        """Unflatten JSON structure"""
        result = {}
        
        for key, value in data.items():
            keys = key.split(separator)
            current = result
            
            for k in keys[:-1]:
                if k not in current:
                    # Check if next key is numeric (list index)
                    next_key = keys[keys.index(k) + 1]
                    current[k] = [] if next_key.isdigit() else {}
                current = current[k]
            
            final_key = keys[-1]
            if final_key.isdigit() and isinstance(current, list):
                # Extend list if necessary
                index = int(final_key)
                while len(current) <= index:
                    current.append(None)
                current[index] = value
            else:
                current[final_key] = value
        
        return result
    
    @staticmethod
    def merge(dict1: Dict[str, Any], dict2: Dict[str, Any], deep: bool = True) -> Dict[str, Any]:
        """Merge two dictionaries"""
        if not deep:
            result = dict1.copy()
            result.update(dict2)
            return result
        
        result = dict1.copy()
        
        for key, value in dict2.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = JSONProcessor.merge(result[key], value, deep=True)
            else:
                result[key] = value
        
        return result
    
    @staticmethod
    def filter_keys(data: Dict[str, Any], 
                   include: Optional[List[str]] = None,
                   exclude: Optional[List[str]] = None) -> Dict[str, Any]:
        """Filter dictionary keys"""
        if include:
            return {k: v for k, v in data.items() if k in include}
        elif exclude:
            return {k: v for k, v in data.items() if k not in exclude}
        else:
            return data.copy()


class DataValidator:
    """Data validation utilities"""
    
    @staticmethod
    def validate_email(email: str) -> bool:
        """Validate email address"""
        pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
        return bool(re.match(pattern, email))
    
    @staticmethod
    def validate_url(url: str) -> bool:
        """Validate URL"""
        pattern = r'^https?://(?:[-\w.])+(?:\:[0-9]+)?(?:/(?:[\w/_.])*(?:\?(?:[\w&=%.])*)?(?:\#(?:[\w.])*)?)?$'
        return bool(re.match(pattern, url))
    
    @staticmethod
    def validate_phone(phone: str) -> bool:
        """Validate phone number"""
        # Remove all non-digit characters
        digits = re.sub(r'\D', '', phone)
        # Check if it's a valid length (10-15 digits)
        return 10 <= len(digits) <= 15
    
    @staticmethod
    def validate_required_fields(data: Dict[str, Any], required_fields: List[str]) -> ValidationResult:
        """Validate required fields"""
        result = ValidationResult(is_valid=True, errors=[], warnings=[])
        
        for field in required_fields:
            if field not in data:
                result.add_error(f"Missing required field: {field}")
            elif data[field] is None or data[field] == "":
                result.add_error(f"Required field '{field}' is empty")
        
        return result
    
    @staticmethod
    def validate_data_types(data: Dict[str, Any], 
                           type_specs: Dict[str, Type]) -> ValidationResult:
        """Validate data types"""
        result = ValidationResult(is_valid=True, errors=[], warnings=[])
        
        for field, expected_type in type_specs.items():
            if field in data:
                if not isinstance(data[field], expected_type):
                    result.add_error(
                        f"Field '{field}' should be {expected_type.__name__}, "
                        f"got {type(data[field]).__name__}"
                    )
        
        return result
    
    @staticmethod
    def validate_ranges(data: Dict[str, Any], 
                       range_specs: Dict[str, Dict[str, Union[int, float]]]) -> ValidationResult:
        """Validate numeric ranges"""
        result = ValidationResult(is_valid=True, errors=[], warnings=[])
        
        for field, range_spec in range_specs.items():
            if field in data and isinstance(data[field], (int, float)):
                value = data[field]
                
                if 'min' in range_spec and value < range_spec['min']:
                    result.add_error(f"Field '{field}' value {value} is below minimum {range_spec['min']}")
                
                if 'max' in range_spec and value > range_spec['max']:
                    result.add_error(f"Field '{field}' value {value} is above maximum {range_spec['max']}")
        
        return result


class SchemaValidator:
    """JSON Schema validator"""
    
    def __init__(self, schema: Dict[str, Any]):
        self.schema = schema
    
    def validate(self, data: Any) -> ValidationResult:
        """Validate data against schema"""
        result = ValidationResult(is_valid=True, errors=[], warnings=[])
        
        try:
            validate(instance=data, schema=self.schema)
        except ValidationError as e:
            result.add_error(f"Schema validation failed: {e.message}")
        except Exception as e:
            result.add_error(f"Validation error: {str(e)}")
        
        return result
    
    @classmethod
    def from_file(cls, schema_file: Union[str, Path]) -> 'SchemaValidator':
        """Create validator from schema file"""
        schema_path = Path(schema_file)
        
        if schema_path.suffix.lower() == '.json':
            with open(schema_path, 'r') as f:
                schema = json.load(f)
        elif schema_path.suffix.lower() in ['.yml', '.yaml']:
            with open(schema_path, 'r') as f:
                schema = yaml.safe_load(f)
        else:
            raise ValueError(f"Unsupported schema file format: {schema_path.suffix}")
        
        return cls(schema)


def encode_base64(data: Union[str, bytes]) -> str:
    """Encode data to base64"""
    if isinstance(data, str):
        data = data.encode('utf-8')
    return base64.b64encode(data).decode('utf-8')


def decode_base64(encoded_data: str) -> bytes:
    """Decode base64 data"""
    return base64.b64decode(encoded_data)


def sanitize_filename(filename: str) -> str:
    """Sanitize filename for safe file system usage"""
    # Remove or replace invalid characters
    sanitized = re.sub(r'[<>:"/\\|?*]', '_', filename)
    
    # Remove leading/trailing dots and spaces
    sanitized = sanitized.strip('. ')
    
    # Limit length
    if len(sanitized) > 255:
        name, ext = Path(sanitized).stem, Path(sanitized).suffix
        max_name_length = 255 - len(ext)
        sanitized = name[:max_name_length] + ext
    
    return sanitized


def deep_merge(*dicts: Dict[str, Any]) -> Dict[str, Any]:
    """Deep merge multiple dictionaries"""
    result = {}
    
    for d in dicts:
        for key, value in d.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = deep_merge(result[key], value)
            else:
                result[key] = value
    
    return result
