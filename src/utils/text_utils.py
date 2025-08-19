
import re
from typing import List, Dict, Set
from collections import Counter
import unicodedata


def clean_text(text: str) -> str:
    """
    Limpiar y normalizar texto
    
    Args:
        text: Texto a limpiar
        
    Returns:
        str: Texto limpio
    """
    if not text:
        return ""
    
    # Normalizar unicode
    text = unicodedata.normalize('NFKD', text)
    
    # Remover caracteres de control y espacios extra
    text = re.sub(r'\s+', ' ', text)  # Múltiples espacios -> un espacio
    text = re.sub(r'\n+', '\n', text)  # Múltiples saltos -> un salto
    text = re.sub(r'[\r\t\f\v]', ' ', text)  # Otros espacios en blanco
    
    # Limpiar caracteres especiales problemáticos
    text = text.replace('\x00', '')  # Null bytes
    text = re.sub(r'[^\x00-\x7F]+', lambda x: x.group(0), text)  # Mantener UTF-8
    
    return text.strip()


def extract_keywords(text: str, max_keywords: int = 10, min_length: int = 3) -> List[str]:
    """
    Extraer palabras clave del texto
    
    Args:
        text: Texto de entrada
        max_keywords: Máximo número de palabras clave
        min_length: Longitud mínima de las palabras
        
    Returns:
        List[str]: Lista de palabras clave
    """
    if not text:
        return []
    
    # Stopwords en español e inglés
    stopwords = {
        # Español
        'el', 'la', 'de', 'que', 'y', 'en', 'un', 'es', 'se', 'no', 'te', 'lo', 'le', 'da', 'su', 'por', 'son', 'con', 'para', 'como', 'las', 'del', 'los', 'una', 'al', 'pero', 'sus', 'ser', 'ha', 'me', 'si', 'sin', 'sobre', 'este', 'ya', 'todo', 'esta', 'cuando', 'muy', 'sin', 'pueden', 'hasta', 'donde', 'quien', 'desde', 'nos', 'durante', 'todos', 'uno', 'les', 'ni', 'contra', 'otros', 'fueron', 'ese', 'eso', 'había', 'ante', 'ellos', 'e', 'esto', 'mí', 'antes', 'algunos', 'qué', 'unos', 'yo', 'otro', 'otras', 'otra', 'él', 'tanto', 'esa', 'estos', 'mucho', 'quienes', 'nada', 'muchos', 'cual', 'poco', 'ella', 'estar', 'estas', 'algunas', 'algo', 'nosotros', 'mi', 'mis', 'tú', 'te', 'ti', 'tu', 'tus', 'ellas', 'nosotras', 'vosotros', 'vosotras', 'os', 'mío', 'mía', 'míos', 'mías', 'tuyo', 'tuya', 'tuyos', 'tuyas', 'suyo', 'suya', 'suyos', 'suyas', 'nuestro', 'nuestra', 'nuestros', 'nuestras', 'vuestro', 'vuestra', 'vuestros', 'vuestras', 'esos', 'esas',
        # Inglés
        'the', 'be', 'to', 'of', 'and', 'a', 'in', 'that', 'have', 'i', 'it', 'for', 'not', 'on', 'with', 'he', 'as', 'you', 'do', 'at', 'this', 'but', 'his', 'by', 'from', 'they', 'we', 'say', 'her', 'she', 'or', 'an', 'will', 'my', 'one', 'all', 'would', 'there', 'their', 'what', 'so', 'up', 'out', 'if', 'about', 'who', 'get', 'which', 'go', 'me', 'when', 'make', 'can', 'like', 'time', 'no', 'just', 'him', 'know', 'take', 'people', 'into', 'year', 'your', 'good', 'some', 'could', 'them', 'see', 'other', 'than', 'then', 'now', 'look', 'only', 'come', 'its', 'over', 'think', 'also', 'back', 'after', 'use', 'two', 'how', 'our', 'work', 'first', 'well', 'way', 'even', 'new', 'want', 'because', 'any', 'these', 'give', 'day', 'most', 'us'
    }
    
    # Tokenizar y limpiar
    words = re.findall(r'\b[a-zA-ZáéíóúñüÁÉÍÓÚÑÜ]+\b', text.lower())
    
    # Filtrar palabras
    filtered_words = [
        word for word in words 
        if len(word) >= min_length and word not in stopwords
    ]
    
    # Contar frecuencias
    word_counts = Counter(filtered_words)
    
    # Devolver las más frecuentes
    keywords = [word for word, count in word_counts.most_common(max_keywords)]
    
    return keywords


def extract_sentences(text: str) -> List[str]:
    """
    Extraer oraciones del texto
    
    Args:
        text: Texto de entrada
        
    Returns:
        List[str]: Lista de oraciones
    """
    if not text:
        return []
    
    # Patrón para dividir oraciones
    sentence_pattern = r'[.!?]+\s+'
    sentences = re.split(sentence_pattern, text)
    
    # Limpiar y filtrar oraciones vacías
    cleaned_sentences = []
    for sentence in sentences:
        sentence = sentence.strip()
        if sentence and len(sentence) > 10:  # Filtrar oraciones muy cortas
            cleaned_sentences.append(sentence)
    
    return cleaned_sentences


def calculate_text_stats(text: str) -> Dict[str, int]:
    """
    Calcular estadísticas del texto
    
    Args:
        text: Texto a analizar
        
    Returns:
        Dict: Estadísticas del texto
    """
    if not text:
        return {
            "characters": 0,
            "characters_no_spaces": 0,
            "words": 0,
            "sentences": 0,
            "paragraphs": 0,
            "avg_word_length": 0,
            "avg_sentence_length": 0
        }
    
    # Contar caracteres
    char_count = len(text)
    char_count_no_spaces = len(text.replace(' ', ''))
    
    # Contar palabras
    words = re.findall(r'\b\w+\b', text)
    word_count = len(words)
    
    # Contar oraciones
    sentences = extract_sentences(text)
    sentence_count = len(sentences)
    
    # Contar párrafos
    paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]
    paragraph_count = len(paragraphs)
    
    # Calcular promedios
    avg_word_length = sum(len(word) for word in words) / word_count if word_count > 0 else 0
    avg_sentence_length = word_count / sentence_count if sentence_count > 0 else 0
    
    return {
        "characters": char_count,
        "characters_no_spaces": char_count_no_spaces,
        "words": word_count,
        "sentences": sentence_count,
        "paragraphs": paragraph_count,
        "avg_word_length": round(avg_word_length, 1),
        "avg_sentence_length": round(avg_sentence_length, 1)
    }


def find_key_phrases(text: str, min_phrase_length: int = 2, max_phrase_length: int = 4) -> List[str]:
    """
    Encontrar frases clave en el texto
    
    Args:
        text: Texto de entrada
        min_phrase_length: Longitud mínima de la frase
        max_phrase_length: Longitud máxima de la frase
        
    Returns:
        List[str]: Frases clave encontradas
    """
    if not text:
        return []
    
    # Stopwords básicas
    stopwords = {'el', 'la', 'de', 'que', 'y', 'en', 'un', 'es', 'se', 'no', 'te', 'lo', 'le', 'da', 'su', 'por', 'son', 'con', 'para', 'the', 'be', 'to', 'of', 'and', 'a', 'in', 'that'}
    
    # Tokenizar en palabras
    words = re.findall(r'\b[a-zA-ZáéíóúñüÁÉÍÓÚÑÜ]+\b', text.lower())
    
    phrases = []
    phrase_counts = Counter()
    
    # Generar n-gramas
    for n in range(min_phrase_length, max_phrase_length + 1):
        for i in range(len(words) - n + 1):
            phrase = words[i:i+n]
            
            # Filtrar frases que empiecen o terminen con stopwords
            if phrase[0] not in stopwords and phrase[-1] not in stopwords:
                phrase_str = ' '.join(phrase)
                phrase_counts[phrase_str] += 1
    
    # Filtrar frases que aparezcan al menos 2 veces
    significant_phrases = [
        phrase for phrase, count in phrase_counts.items() 
        if count >= 2
    ]
    
    # Ordenar por frecuencia
    significant_phrases.sort(key=lambda x: phrase_counts[x], reverse=True)
    
    return significant_phrases[:20]  # Top 20 frases


def highlight_terms(text: str, terms: List[str], highlight_tag: str = "**") -> str:
    """
    Resaltar términos en el texto
    
    Args:
        text: Texto original
        terms: Términos a resaltar
        highlight_tag: Tag para resaltar (ej: "**" para markdown bold)
        
    Returns:
        str: Texto con términos resaltados
    """
    if not text or not terms:
        return text
    
    highlighted_text = text
    
    for term in terms:
        # Escapar caracteres especiales de regex
        escaped_term = re.escape(term)
        pattern = r'\b' + escaped_term + r'\b'
        
        # Resaltar término (case insensitive)
        highlighted_text = re.sub(
            pattern, 
            f"{highlight_tag}{term}{highlight_tag}", 
            highlighted_text, 
            flags=re.IGNORECASE
        )
    
    return highlighted_text


def truncate_text(text: str, max_length: int = 500, suffix: str = "...") -> str:
    """
    Truncar texto a una longitud máxima
    
    Args:
        text: Texto a truncar
        max_length: Longitud máxima
        suffix: Sufijo a añadir si se trunca
        
    Returns:
        str: Texto truncado
    """
    if not text or len(text) <= max_length:
        return text
    
    # Truncar en la palabra más cercana al límite
    truncated = text[:max_length]
    last_space = truncated.rfind(' ')
    
    if last_space > max_length * 0.8:  # Si hay un espacio cercano al final
        truncated = truncated[:last_space]
    
    return truncated + suffix


def similarity_score(text1: str, text2: str) -> float:
    """
    Calcular similitud simple entre dos textos usando Jaccard
    
    Args:
        text1: Primer texto
        text2: Segundo texto
        
    Returns:
        float: Score de similitud entre 0 y 1
    """
    if not text1 or not text2:
        return 0.0
    
    # Tokenizar y convertir a conjuntos
    words1 = set(re.findall(r'\b\w+\b', text1.lower()))
    words2 = set(re.findall(r'\b\w+\b', text2.lower()))
    
    # Calcular similitud de Jaccard
    intersection = len(words1.intersection(words2))
    union = len(words1.union(words2))
    
    return intersection / union if union > 0 else 0.0