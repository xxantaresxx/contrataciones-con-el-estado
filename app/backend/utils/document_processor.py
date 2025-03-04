import os
from typing import List, Dict, Any
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import tiktoken
from tqdm import tqdm
from pathlib import Path
import pickle
import hashlib

class DocumentProcessor:
    def __init__(self, docs_path: str):
        self.docs_path = Path(docs_path).resolve()
        if not self.docs_path.exists():
            raise ValueError(f"El directorio {self.docs_path} no existe")
        
        self.chunk_size = 1000
        self.chunk_overlap = 200
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            length_function=len
        )
        # Usar un modelo más ligero optimizado para español
        self.embedding_model = SentenceTransformer('distiluse-base-multilingual-cased-v2')
        self.embeddings = None
        self.chunks = []
        self.chunk_metadata = []
        
        # Crear directorio para caché si no existe
        self.cache_dir = self.docs_path.parent / "cache"
        self.cache_dir.mkdir(exist_ok=True)

    def _get_cache_key(self) -> str:
        """Genera una clave de caché basada en los archivos y sus contenidos."""
        pdf_files = sorted(self.docs_path.glob("*.pdf"))
        hash_content = ""
        for pdf in pdf_files:
            hash_content += f"{pdf.name}_{pdf.stat().st_mtime}"
        return hashlib.md5(hash_content.encode()).hexdigest()

    def _load_from_cache(self, cache_key: str) -> bool:
        """Intenta cargar los datos desde la caché."""
        cache_file = self.cache_dir / f"{cache_key}.pkl"
        if cache_file.exists():
            try:
                print("Cargando datos desde caché...")
                with open(cache_file, 'rb') as f:
                    cached_data = pickle.load(f)
                    self.chunks = cached_data['chunks']
                    self.chunk_metadata = cached_data['metadata']
                    self.embeddings = cached_data['embeddings']
                print("Datos cargados exitosamente desde caché")
                return True
            except Exception as e:
                print(f"Error al cargar caché: {e}")
        return False

    def _save_to_cache(self, cache_key: str):
        """Guarda los datos procesados en caché."""
        cache_file = self.cache_dir / f"{cache_key}.pkl"
        try:
            print("Guardando datos en caché...")
            with open(cache_file, 'wb') as f:
                pickle.dump({
                    'chunks': self.chunks,
                    'metadata': self.chunk_metadata,
                    'embeddings': self.embeddings
                }, f)
            print("Datos guardados exitosamente en caché")
        except Exception as e:
            print(f"Error al guardar caché: {e}")

    def process_pdf(self, file_path: str) -> List[str]:
        """Procesa un archivo PDF y retorna una lista de chunks de texto."""
        try:
            print(f"Procesando archivo: {file_path}")
            reader = PdfReader(file_path)
            text = ""
            for page in reader.pages:
                text += page.extract_text()
            
            chunks = self.text_splitter.split_text(text)
            print(f"Se extrajeron {len(chunks)} chunks del archivo")
            return chunks
        except Exception as e:
            print(f"Error procesando {file_path}: {str(e)}")
            return []

    def process_all_documents(self):
        """Procesa todos los documentos PDF en el directorio especificado."""
        # Verificar caché primero
        cache_key = self._get_cache_key()
        if self._load_from_cache(cache_key):
            return

        self.chunks = []
        self.chunk_metadata = []
        
        # Listar todos los archivos PDF en el directorio
        pdf_files = list(self.docs_path.glob("*.pdf"))
        if not pdf_files:
            raise ValueError(f"No se encontraron archivos PDF en {self.docs_path}")
        
        print(f"Encontrados {len(pdf_files)} archivos PDF")
        for pdf_file in pdf_files:
            print(f"Procesando: {pdf_file.name}")
            chunks = self.process_pdf(str(pdf_file))
            
            for chunk in chunks:
                self.chunks.append(chunk)
                self.chunk_metadata.append({
                    'source': pdf_file.name,
                    'text': chunk[:100] + '...'  # Preview del chunk
                })
        
        print(f"Total de chunks procesados: {len(self.chunks)}")
        
        # Generar embeddings inmediatamente
        if self.chunks:
            print("Generando embeddings...")
            self.create_embeddings()
            # Guardar en caché
            self._save_to_cache(cache_key)

    def create_embeddings(self):
        """Crea embeddings para todos los chunks."""
        if not self.chunks:
            raise ValueError("No hay chunks para procesar")

        print("Generando embeddings...")
        # Procesar chunks en lotes para mejor rendimiento
        batch_size = 32
        embeddings = []
        
        for i in tqdm(range(0, len(self.chunks), batch_size)):
            batch = self.chunks[i:i + batch_size]
            batch_embeddings = self.embedding_model.encode(batch, show_progress_bar=False)
            embeddings.extend(batch_embeddings)

        self.embeddings = np.array(embeddings)
        print(f"Embeddings creados para {len(self.chunks)} chunks")

    def search_similar_chunks(self, query: str, k: int = 5) -> List[Dict[str, Any]]:
        """Busca los chunks más similares a una consulta usando similitud coseno."""
        if self.embeddings is None:
            raise ValueError("Los embeddings no están inicializados")

        # Generar embedding para la consulta
        query_vector = self.embedding_model.encode([query])
        
        # Calcular similitud coseno
        similarities = cosine_similarity(query_vector, self.embeddings)[0]
        
        # Obtener los k índices más similares
        top_k_indices = similarities.argsort()[-k:][::-1]
        
        results = []
        for idx in top_k_indices:
            results.append({
                'chunk': self.chunks[idx],
                'metadata': self.chunk_metadata[idx],
                'score': float(similarities[idx])
            })
        
        return results

    def get_token_count(self, text: str) -> int:
        """Cuenta el número de tokens en un texto usando el tokenizer de GPT-4."""
        encoding = tiktoken.encoding_for_model("gpt-4")
        return len(encoding.encode(text)) 