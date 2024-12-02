import streamlit as st
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
import openai
import os

# Configura tu clave de OpenAI desde las variables de entorno
openai.api_key = os.getenv("OPENAI_API_KEY")

# Función para cargar y procesar PDFs
def procesar_pdfs(directorio):
    loader = PyPDFLoader(directorio)
    documents = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    return text_splitter.split_documents(documents)

# Carga los documentos y crea el índice
@st.cache_data
def crear_indice(directorio):
    texts = procesar_pdfs(directorio)
    embeddings = OpenAIEmbeddings()
    vector_store = FAISS.from_documents(texts, embeddings)
    return vector_store

# Interfaz de Streamlit
st.title("Asistente Inteligente de PDFs 📄")
st.sidebar.title("Configuración")
directorio = st.sidebar.text_input("Ruta de los PDFs", "./")  # Cambia "./" por tu carpeta de PDFs en la nube
consultar = st.sidebar.button("Crear Índice")

if consultar:
    st.info("Procesando documentos y creando índice...")
    try:
        vector_store = crear_indice(directorio)
        st.success("¡Índice creado con éxito!")
    except Exception as e:
        st.error(f"Error al procesar los PDFs: {e}")

# Interfaz para preguntas
if "vector_store" in locals() or "vector_store" in globals():
    consulta = st.text_input("Haz tu pregunta:", key="consulta")
    if consulta:
        results = vector_store.similarity_search(consulta, k=3)
        for i, result in enumerate(results):
            st.write(f"**Resultado {i+1}:** {result.page_content}")
else:
    st.warning("Por favor, crea un índice antes de realizar consultas.")
