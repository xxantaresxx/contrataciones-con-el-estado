# Ley Contrataciones con el Estado

Este proyecto es una aplicación que utiliza la API de OpenAI para responder preguntas relacionadas con las contrataciones públicas en Perú.

## Requisitos

- Python 3.8 o superior
- pip

## Instalación

1. Clona el repositorio:

   ```bash
   git clone https://github.com/tu_usuario/nombre_del_repositorio.git
   cd nombre_del_repositorio
   ```

2. Crea un entorno virtual:

   ```bash
   python -m venv venv
   ```

3. Activa el entorno virtual:

   - En Windows:

     ```bash
     .\venv\Scripts\Activate.ps1
     ```

   - En macOS/Linux:

     ```bash
     source venv/bin/activate
     ```

4. Instala las dependencias:

   ```bash
   pip install -r requirements.txt
   ```

5. Configura las variables de entorno:

   Crea un archivo `.env` en la raíz del proyecto y agrega tu clave de API de OpenAI:

   ```plaintext
   OPENAI_API_KEY=tu_clave_de_api
   ```

## Uso

Para ejecutar la aplicación, utiliza el siguiente comando:

```bash
uvicorn app.main:app --reload
```

## Contribuciones

Las contribuciones son bienvenidas. Siéntete libre de abrir un issue o un pull request. 