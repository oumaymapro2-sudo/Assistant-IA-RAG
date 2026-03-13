import streamlit as st
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from config import CHUNK_SIZE, CHUNK_OVERLAP, TEMPERATURE, TOP_K_DOCUMENTS, MODEL_NAME

import os
from pathlib import Path

# Configuration page
st.set_page_config(
    page_title="RAG Documentation Assistant",
    page_icon="C:\\Projects-lj-WebData\\IA-project\\rag_technical_docs\\public\\favicon.ico",
    layout="wide"
)

st.title("RAG Technical Documentation Assistant")
st.markdown("*Système de question-réponse sur documentation technique*")

# Sidebar
with st.sidebar:
    st.header("Configuration")
    
    api_key = st.text_input(
        "OpenAI API Key",
        type="password",
        help="Votre clé API OpenAI"
    )
    
    if api_key:
        os.environ["OPENAI_API_KEY"] = api_key
        st.success("API Key configurée")
    
    st.divider()
    
    st.markdown("""
    ### Stack utilisée
    - LangChain
    - OpenAI GPT-3.5
    - FAISS
    - Streamlit
    
    **Développé par :**  
    Oumayma Lamjar  
    LJ WebData | Data Scientist & IA

    """)

# Chargement des documents
@st.cache_resource
def load_documents():
    data_path = Path("data")
    
    if not data_path.exists():
        st.error("Le dossier 'data/' n'existe pas")
        return None
    
    try:
        loader = DirectoryLoader(
            str(data_path),
            glob="**/*.pdf",
            loader_cls=PyPDFLoader
        )
        docs = loader.load()
        
        if not docs:
            st.warning("Aucun PDF trouvé")
            return None
            
        return docs
    except Exception as e:
        st.error(f"Erreur: {e}")
        return None

# Création de la base vectorielle
@st.cache_resource
def create_vectorstore(_documents):
    # Découpage en chunks
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP
    )
    chunks = splitter.split_documents(_documents)
    
    # Embeddings et vectorisation
    embeddings = OpenAIEmbeddings()
    vectorstore = FAISS.from_documents(chunks, embeddings)
    
    return vectorstore

# Pipeline RAG
def create_rag_chain(vectorstore):
    llm = ChatOpenAI(model=MODEL_NAME, temperature=TEMPERATURE)
    
    # Retriever pour chercher dans les docs
    retriever = vectorstore.as_retriever(search_kwargs={"k": TOP_K_DOCUMENTS})
    
    # Prompt template
    system_prompt = """Tu es un assistant qui répond aux questions basées sur la documentation fournie.
    Si tu ne sais pas, dis-le clairement.
    
    Contexte:
    {context}
    """
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "{input}")
    ])
    
    # Création de la chain
    qa_chain = create_stuff_documents_chain(llm, prompt)
    rag_chain = create_retrieval_chain(retriever, qa_chain)
    
    return rag_chain

# Main
if not api_key:
    st.info("Entrez votre clé API OpenAI dans la sidebar")
    st.stop()

# Chargement et indexation
with st.spinner("Chargement des documents..."):
    if 'vectorstore' not in st.session_state:
        documents = load_documents()
        
        if documents:
            st.session_state.doc_count = len(documents)
            vectorstore = create_vectorstore(documents)
            st.session_state.vectorstore = vectorstore
            st.session_state.rag_chain = create_rag_chain(vectorstore)
            st.success(f"{len(documents)} documents indexés")

# Interface chat
if 'vectorstore' in st.session_state:
    st.subheader("Posez vos questions")
    
    # Historique des messages
    if 'messages' not in st.session_state:
        st.session_state.messages = []
    
    # Afficher l'historique
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
    
    # Input utilisateur
    if question := st.chat_input("Ex: Comment fonctionne Arduino ?"):
        # Afficher la question
        st.session_state.messages.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.markdown(question)
        
        # Générer la réponse
        with st.chat_message("assistant"):
            with st.spinner("Recherche..."):
                response = st.session_state.rag_chain.invoke({"input": question})
                answer = response["answer"]
                
                st.markdown(answer)
                
                # Afficher les sources
                with st.expander("Sources"):
                    for i, doc in enumerate(response["context"], 1):
                        source_name = Path(doc.metadata.get("source", "")).name
                        st.markdown(f"**{i}.** {source_name}")
                        st.text(doc.page_content[:120] + "...")
                
                st.session_state.messages.append({"role": "assistant", "content": answer})
    
    # Reset conversation
    if st.button("Nouvelle conversation"):
        st.session_state.messages = []
        st.rerun()
