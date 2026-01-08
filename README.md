# Asistente Personal de "Segundo Cerebro" Basado en LLM
Este proyecto consiste en un asistente personal inteligente diseñado para actuar como un "Segundo Cerebro", utilizando Modelos de Lenguaje de Gran Escala (LLM) para gestionar, recuperar y organizar información personal de diversas fuentes.


🧠 Descripción del Proyecto
El sistema permite al usuario interactuar con su propia base de conocimientos, que incluye apuntes de Notion y eventos de Google Calendar, facilitando la toma de decisiones y la gestión de información mediante una interfaz conversacional.



🏗️ Arquitectura del Sistema
El asistente se basa en una arquitectura de agentes especializados que trabajan de forma coordinada:

Agente de Diálogo: Gestiona la interacción natural con el usuario. Desarrollado con LangChain y desplegado con Streamlit Cloud.

Agente de Recuperación (RAG): Extrae información relevante de fuentes como archivos de texto (exportaciones de Notion) y el calendario personal. Utiliza tres agentes RAG, incluyendo uno específico para Google Calendar.

Agente de Organización: Estructura la información mediante el particionamiento de archivos en chunks y su codificación en embeddings a través de OpenAI, almacenándolos en una base de datos vectorial FAISS.

Agente de Generación: Utiliza GPT para crear respuestas, ideas y soluciones basadas en la información recuperada.

Agente de Personalización: Adapta el estilo y contenido de las respuestas mediante un prompt específico según los objetivos del usuario.


🛠️ Tecnologías Utilizadas

Framework de LLM: LangChain.

Modelo de Lenguaje: OpenAI GPT.

Base de Datos Vectorial: FAISS.

Embeddings: OpenAI Embeddings.

Interfaz de Usuario (Frontend): Streamlit.

Integraciones: Google Calendar API y exportaciones de Notion (archivos de texto).


🚀 Funcionalidades Principales

Gestión de tareas: Control de recordatorios y pendientes.


Búsqueda personal: Recuperación eficiente de información contenida en apuntes (ej. apuntes del curso de R en Notion).


Generación creativa: Creación de ideas y soluciones basadas en el contexto del usuario.

Síntesis: Resumen de información compleja acumulada.

Planificación: Programación y consulta de eventos en el calendario.


🚀 Demo y Documentación
Acceso al Asistente en Vivo:
https://codigofacilito-esvrahlvzd8m44ilquyqbj.streamlit.app/

Descripción Detallada del Proyecto (PDF):
https://github.com/jeremyvillanuevar/codigofacilito/blob/main/Proyecto%20Segundo%20Cerebro.pdf

📌 Backlog (Próximas Mejoras)
Agente OCR: Incorporar la revisión de imágenes.

Soporte Multimedia: Agregar procesamiento de video.

Optimización: Incrementar la eficiencia en la búsqueda de notas.

Flexibilidad de Modelos: Permitir al usuario intercambiar el modelo de lenguaje (LLM) utilizado.


Por favor espero feedback,

Saludos,
Jeremy
