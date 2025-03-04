import os
from typing import List, Dict, Any
from openai import OpenAI
from dotenv import load_dotenv
from fastapi import HTTPException

# Cargar variables de entorno
load_dotenv()

class GPTHandler:
    def __init__(self, use_gpt4: bool = False):
        api_key = os.getenv('OPENAI_API_KEY')
        if not api_key:
            raise ValueError("OPENAI_API_KEY no encontrada en variables de entorno")
        
        # Inicializar el cliente de OpenAI
        self.client = OpenAI(api_key=api_key)
        
        # Seleccionar el modelo (usando GPT-4)
        self.model = "gpt-4" if use_gpt4 else "gpt-3.5-turbo"
        
        self.system_prompt = """Eres un asistente legal especializado en contrataciones públicas del Perú.
        Tu tarea es responder preguntas basadas en la documentación proporcionada.
        Debes:
        1. Proporcionar respuestas precisas y fundamentadas en los documentos.
        2. Citar las fuentes específicas de donde obtienes la información.
        3. Indicar cuando no tengas suficiente información para responder.
        4. Mantener un tono profesional y objetivo."""

    async def get_response(
        self,
        query: str,
        relevant_chunks: List[Dict[str, Any]],
        max_tokens: int = 1000
    ) -> Dict[str, Any]:
        """
        Obtiene una respuesta usando OpenAI basada en los chunks relevantes.
        """
        try:
            # Preparar el contexto con los chunks relevantes
            context = "\n\n".join([
                f"Documento: {chunk['metadata']['source']}\n{chunk['chunk']}"
                for chunk in relevant_chunks
            ])

            # Crear el mensaje para el modelo
            stream = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": f"""
                    Contexto relevante:
                    {context}

                    Pregunta del usuario:
                    {query}

                    Por favor, responde la pregunta basándote en el contexto proporcionado.
                    Incluye referencias a los documentos específicos cuando sea posible."""}
                ],
                max_tokens=max_tokens,
                temperature=0.1,
                stream=True
            )

            # Recopilar la respuesta completa del stream
            full_response = ""
            for chunk in stream:
                if chunk.choices[0].delta.content:
                    full_response += chunk.choices[0].delta.content

            # Calcular confianza basada en los scores de los chunks
            confidence = sum(chunk['score'] for chunk in relevant_chunks) / len(relevant_chunks)

            # Crear la jerarquía de artículos
            article_hierarchy = []
            for chunk in relevant_chunks:
                if 'article' in chunk['metadata']:
                    article_hierarchy.append(chunk['metadata']['article'])
            article_hierarchy = list(set(article_hierarchy))  # Eliminar duplicados

            return {
                "answer": full_response,
                "confidence": confidence,
                "sources": article_hierarchy
            }

        except Exception as e:
            error_msg = f"Error al comunicarse con OpenAI: {str(e)}"
            print(f"Error: {error_msg}")
            raise HTTPException(status_code=503, detail=error_msg)

    def generate_follow_up_questions(self, context: str, previous_qa: List[Dict[str, str]]) -> List[str]:
        """
        Genera preguntas de seguimiento usando OpenAI.
        """
        try:
            stream = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": """
                    Genera 3 preguntas de seguimiento relevantes basadas en el contexto y las preguntas anteriores.
                    Las preguntas deben:
                    1. Profundizar en aspectos importantes no cubiertos
                    2. Aclarar puntos que podrían ser confusos
                    3. Explorar implicaciones prácticas
                    """},
                    {"role": "user", "content": f"""
                    Contexto: {context}

                    Preguntas y respuestas anteriores:
                    {previous_qa}

                    Genera 3 preguntas de seguimiento relevantes."""}
                ],
                temperature=0.1,
                max_tokens=1000,
                stream=True
            )

            # Recopilar la respuesta completa del stream
            full_response = ""
            for chunk in stream:
                if chunk.choices[0].delta.content:
                    full_response += chunk.choices[0].delta.content

            # Procesar la respuesta para extraer las preguntas
            questions = [q.strip() for q in full_response.split("\n") if q.strip()]
            return questions[:3]  # Asegurar que solo devolvemos 3 preguntas

        except Exception as e:
            print(f"Error al generar preguntas de seguimiento: {str(e)}")
            return [] 