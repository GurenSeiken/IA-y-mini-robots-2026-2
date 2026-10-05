import os
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

def create_rag_curso(docs_dir="documentos_curso", persist_directory="chroma_db_curso"):
    # 1. Cargar documentos
    if not os.path.exists(docs_dir):
        os.makedirs(docs_dir)
        print(f"Directorio '{docs_dir}' creado. Por favor, añade los documentos del curso (TXTs) aquí.")
        return None

    print("Cargando documentos del curso...")
    loader = DirectoryLoader(docs_dir, glob="**/*.txt", loader_cls=TextLoader, loader_kwargs={'encoding': 'utf-8'})
    documents = loader.load()
    
    if not documents:
        print(f"No se encontraron documentos en {docs_dir}. Añade TXTs para continuar.")
        return None

    # 2. Dividir documentos en chunks
    print("Procesando documentos...")
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    texts = text_splitter.split_documents(documents)

    # 3. Crear embeddings y base de datos vectorial
    print("Creando base de datos vectorial...")
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vectordb = Chroma.from_documents(documents=texts, embedding=embeddings, persist_directory=persist_directory)

    # 4. Inicializar LLM (usando LM Studio)
    # Se usa ChatOpenAI apuntando al servidor local de LM Studio
    print("Inicializando modelo de lenguaje (LM Studio - Gemma 2 2B)...")
    llm = ChatOpenAI(base_url="http://localhost:1234/v1", api_key="lm-studio", temperature=0)

    # 5. Crear la cadena RAG usando LCEL (LangChain Expression Language)
    retriever = vectordb.as_retriever(search_kwargs={"k": 3})
    
    template = """Usa la siguiente información para responder a la pregunta.
    Si no sabes la respuesta, simplemente di que no lo sabes.
    
    Contexto: {context}
    
    Pregunta: {question}
    Respuesta:"""
    prompt = PromptTemplate.from_template(template)
    
    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)
        
    qa_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    
    return qa_chain, retriever

if __name__ == "__main__":
    result = create_rag_curso()
    if result:
        qa_chain, retriever = result
        print("\n¡Chatbot del Curso listo! Escribe 'salir' para terminar.")
        while True:
            query = input("Estudiante: ")
            if query.lower() in ["salir", "exit", "quit"]:
                break
            
            response = qa_chain.invoke(query)
            print(f"\nChatbot: {response}\n")
            
            docs = retriever.invoke(query)
            print("Fuentes consultadas:")
            for doc in docs:
                print(f"- {doc.metadata.get('source', 'Desconocido')}")
            print("-" * 50)
