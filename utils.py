# utils.py

import streamlit as st
import openai
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferWindowMemory
from langchain.chat_models import ChatOpenAI
from langchain.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings
from langchain.prompts import PromptTemplate
from langchain.prompts.chat import SystemMessagePromptTemplate

openai.api_key = st.secrets["OPENAI_API_KEY"]

@st.cache_resource
def load_chain():
  """
    The `load_chain()` function initializes and configures a conversational retrieval chain for
    answering user questions.
    :return: The `load_chain()` function returns a ConversationalRetrievalChain object.
    """

  # Load OpenAI embedding model
  embeddings = OpenAIEmbeddings()
  
  # Load OpenAI chat model
  llm = ChatOpenAI(temperature=0)
  
  # Load our local FAISS index as a retriever
  vector_store = FAISS.load_local("faiss_index", embeddings,allow_dangerous_deserialization=True)
  retriever = vector_store.as_retriever(search_kwargs={"k": 3})
  
  # Create memory 'chat_history' 
  memory = ConversationBufferWindowMemory(k=3,memory_key="chat_history")
  
  # Create system prompt
  template = """
        Eres un asistente de AI con las siguientes funcionalidades principales:
1. Gestión de tareas y recordatorios. (cuando te pregunten que hay agendado el dia de hoy por ejemplo utiliza la informacion que tienes de mi Calendario de Google como contexto)
2. Búsqueda y recuperación de información personal (responde preguntas sobre la base de datos de notas que hice en mis clases de Data Science con el lenguaje de programación R, curso que tome en Abril aproximadamente de este año 2024 brindado como contexto)
3. Generación de ideas y soluciones creativas
4. Resumen y síntesis de información.
5. Planificación y programación de eventos.
        Adicionalmente al contexto te brindare una pregunta.
        Por favor pon una respuesta conversacional, y si no sabes la respuesta dices 'Disculpa, No lo se...', no trates de crear una respuesta.
        Si la pregunta esta fuera de tu funcionalidad, por favor con cortesía informa que solo daras respuestas a preguntas a las funcionalidades que te diseñamos.
        Aquí esta la información de base de datos de notas que hice en mis clases de Data Science con el lenguaje de programación R y mi Calendario de Google:
    {context}
    Question: {question}
    Helpful Answer:"""
  
  # Create the Conversational Chain
  chain = ConversationalRetrievalChain.from_llm(llm=llm, 
                                              retriever=retriever, 
                                              memory=memory, 
                                              get_chat_history=lambda h : h,
                                              verbose=True)
  
  # Add systemp prompt to chain
  # Can only add it at the end for ConversationalRetrievalChain
  QA_CHAIN_PROMPT = PromptTemplate(input_variables=["context", "question"],template=template)
  chain.combine_docs_chain.llm_chain.prompt.messages[0] = SystemMessagePromptTemplate(prompt=QA_CHAIN_PROMPT)
  
  return chain