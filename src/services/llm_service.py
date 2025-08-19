
import openai
from typing import List, Dict, Any, Optional
from abc import ABC, abstractmethod

from config.settings import settings


class BaseLLMService(ABC):
    
    @abstractmethod
    def generate_response(self, prompt: str, max_tokens: int = 1000) -> str:
        pass
    
    @abstractmethod
    def generate_summary(self, text: str, max_length: int = 200) -> str:
        pass


class OpenAIService(BaseLLMService):
    
    def __init__(self, api_key: str, model: str = "gpt-3.5-turbo"):
        self.client = openai.OpenAI(api_key=api_key)
        self.model = model
    
    def generate_response(self, prompt: str, max_tokens: int = 1000) -> str:
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "Eres un asistente experto en análisis de documentos. Responde de manera clara, precisa y estructurada."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=max_tokens,
                temperature=0.7
            )
            
            return response.choices[0].message.content.strip()
            
        except Exception as e:
            return f"Error generando respuesta: {str(e)}"
    
    def generate_summary(self, text: str, max_length: int = 200) -> str:
        prompt = f"""Resume el siguiente texto en máximo {max_length} palabras, 
        capturando los puntos más importantes:

        {text}

        Resumen:"""
        
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "Eres un experto en crear resúmenes concisos y precisos."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=max_length + 50,
                temperature=0.3
            )
            
            return response.choices[0].message.content.strip()
            
        except Exception as e:
            return f"Error generando resumen: {str(e)}"
    
    def analyze_topics(self, text: str) -> List[str]:
        prompt = f"""Identifica los 5 temas principales del siguiente texto.
        Responde solo con una lista de temas separados por comas:

        {text[:2000]}  # Limitar texto para evitar tokens excesivos

        Temas:"""
        
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "Eres un experto en análisis temático de documentos."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=100,
                temperature=0.3
            )
            
            topics_text = response.choices[0].message.content.strip()
            topics = [topic.strip() for topic in topics_text.split(',')]
            return topics[:5]  # Máximo 5 temas
            
        except Exception as e:
            return ["Error analizando temas"]


class AnthropicService(BaseLLMService):
    
    def __init__(self, api_key: str, model: str = "claude-3-sonnet-20240229"):
        try:
            import anthropic
            self.client = anthropic.Anthropic(api_key=api_key)
            self.model = model
        except ImportError:
            raise ImportError("Instala la librería anthropic: pip install anthropic")
    
    def generate_response(self, prompt: str, max_tokens: int = 1000) -> str:
        try:
            response = self.client.messages.create(
                model=self.model,
                max_tokens=max_tokens,
                temperature=0.7,
                system="Eres un asistente experto en análisis de documentos. Responde de manera clara, precisa y estructurada.",
                messages=[
                    {"role": "user", "content": prompt}
                ]
            )
            
            return response.content[0].text.strip()
            
        except Exception as e:
            return f"Error generando respuesta: {str(e)}"
    
    def generate_summary(self, text: str, max_length: int = 200) -> str:
        """Generar resumen del texto"""
        prompt = f"""Resume el siguiente texto en máximo {max_length} palabras, 
        capturando los puntos más importantes:

        {text}"""
        
        try:
            response = self.client.messages.create(
                model=self.model,
                max_tokens=max_length + 50,
                temperature=0.3,
                system="Eres un experto en crear resúmenes concisos y precisos.",
                messages=[
                    {"role": "user", "content": prompt}
                ]
            )
            
            return response.content[0].text.strip()
            
        except Exception as e:
            return f"Error generando resumen: {str(e)}"


class MockLLMService(BaseLLMService):
    
    def generate_response(self, prompt: str, max_tokens: int = 1000) -> str:
        return f"Respuesta simulada para: {prompt[:100]}..."
    
    def generate_summary(self, text: str, max_length: int = 200) -> str:
        return f"Resumen simulado del texto de {len(text)} caracteres."


class LLMService:
    
    def __init__(self):
        self.service = self._initialize_service()
    
    def _initialize_service(self) -> BaseLLMService:
        if settings.openai_api_key:
            return OpenAIService(settings.openai_api_key)
        elif settings.anthropic_api_key:
            return AnthropicService(settings.anthropic_api_key)
        else:
            print("⚠️  No se encontraron API keys. Usando servicio mock.")
            return MockLLMService()
    
    def generate_response(self, prompt: str, max_tokens: int = 1000) -> str:
        return self.service.generate_response(prompt, max_tokens)
    
    def generate_summary(self, text: str, max_length: int = 200) -> str:
        return self.service.generate_summary(text, max_length)
    
    def analyze_topics(self, text: str) -> List[str]:
        if hasattr(self.service, 'analyze_topics'):
            return self.service.analyze_topics(text)
        else:
            # Fallback simple para servicios que no implementen análisis de temas
            words = text.lower().split()
            common_words = ['el', 'la', 'de', 'que', 'y', 'en', 'un', 'es', 'se', 'no', 'te', 'lo', 'le', 'da', 'su', 'por', 'son', 'con', 'para']
            filtered_words = [w for w in words if len(w) > 4 and w not in common_words]
            
            # Contar frecuencias
            word_freq = {}
            for word in filtered_words:
                word_freq[word] = word_freq.get(word, 0) + 1
            
            # Devolver las 5 palabras más frecuentes como temas
            top_words = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)[:5]
            return [word for word, freq in top_words]