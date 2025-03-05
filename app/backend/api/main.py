from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from contextlib import asynccontextmanager
from pydantic import BaseModel
from typing import List, Optional
import os
import sys
from pathlib import Path
from dotenv import load_dotenv
import asyncio

# Añadir el directorio raíz al path de Python
ROOT_DIR = Path(__file__).resolve().parent.parent.parent.parent
sys.path.append(str(ROOT_DIR))

from app.backend.utils.document_processor import DocumentProcessor
from app.backend.utils.gpt_handler import GPTHandler

# Cargar variables de entorno
load_dotenv()

# Configurar directorios
REGLAMENTOS_DIR = ROOT_DIR / "reglamentos"
STATIC_DIR = Path(__file__).resolve().parent.parent / "static"
CACHE_DIR = ROOT_DIR / "cache"

# Crear directorios necesarios
for directory in [REGLAMENTOS_DIR, STATIC_DIR, CACHE_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

print(f"\nDirectorios del sistema:")
print(f"- Directorio raíz: {ROOT_DIR}")
print(f"- Directorio de documentos: {REGLAMENTOS_DIR}")
print(f"- Directorio estático: {STATIC_DIR}")
print(f"- Directorio de caché: {CACHE_DIR}\n")

# Variables globales para los procesadores
doc_processor = None
gpt_handler = None

async def initialize_processors():
    """Inicializa los procesadores de forma asíncrona."""
    global doc_processor, gpt_handler
    
    try:
        print("\nInicializando procesadores...")
        
        # Verificar documentos PDF
        pdf_files = list(REGLAMENTOS_DIR.glob("*.pdf"))
        if not pdf_files:
            print("\nADVERTENCIA: No se encontraron archivos PDF.")
            print(f"Por favor, coloque sus documentos PDF en: {REGLAMENTOS_DIR}")
            return
        
        print(f"\nDocumentos encontrados ({len(pdf_files)}):")
        for pdf in pdf_files:
            print(f"- {pdf.name}")
        
        # Inicializar procesadores
        print("\nInicializando Document Processor...")
        doc_processor = DocumentProcessor(docs_path=str(REGLAMENTOS_DIR))
        
        print("\nInicializando GPT Handler...")
        use_gpt4 = os.getenv("USE_GPT4", "false").lower() == "true"
        gpt_handler = GPTHandler(use_gpt4=use_gpt4)
        print(f"Usando modelo: {'GPT-4' if use_gpt4 else 'GPT-3.5'}")
        
        # Procesar documentos (esto incluirá la generación de embeddings)
        await asyncio.get_event_loop().run_in_executor(None, doc_processor.process_all_documents)
        print("\nSistema inicializado y listo para procesar consultas")
        
    except Exception as e:
        print(f"\nError durante la inicialización: {str(e)}")
        print("El sistema continuará funcionando con funcionalidad limitada")

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Inicializar al arrancar
    await initialize_processors()
    yield
    # Limpiar al cerrar
    print("\nLimpiando recursos...")

# Inicializar FastAPI
app = FastAPI(
    title="Sistema de Análisis de Documentos Legales",
    description="API para análisis de documentos legales usando LLMs y técnicas de NLP",
    version="1.0.0",
    lifespan=lifespan
)

# Configurar CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Montar archivos estáticos
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

# Modelos Pydantic
class Query(BaseModel):
    question: str
    context: Optional[str] = None

class Response(BaseModel):
    answer: str
    confidence: float
    sources: List[str]

# Rutas
@app.get("/")
async def root():
    return FileResponse(str(STATIC_DIR / "index.html"))

@app.post("/query", response_model=Response)
async def process_query(query: Query):
    if not doc_processor or not doc_processor.chunks:
        raise HTTPException(
            status_code=503,
            detail="Sistema inicializando o sin documentos. Por favor, espere unos momentos o verifique que existan documentos PDF."
        )

    try:
        # Buscar chunks relevantes
        relevant_chunks = doc_processor.search_similar_chunks(query.question)
        if not relevant_chunks:
            raise HTTPException(
                status_code=404,
                detail="No se encontraron documentos relevantes. Por favor, reformule su pregunta."
            )
        
        # Obtener respuesta del modelo
        response = await gpt_handler.get_response(query.question, relevant_chunks)
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        print(f"Error procesando consulta: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail="Error interno del servidor. Por favor, intente nuevamente."
        )

def start():
    """Función para iniciar el servidor"""
    import uvicorn
    host = os.getenv("HOST", "127.0.0.1")
    port = int(os.getenv("PORT", "8000"))
    uvicorn.run(
        "app.backend.api.main:app",
        host=host,
        port=port,
        reload=True
    )

if __name__ == "__main__":
    start() 