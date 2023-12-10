import streamlit as st
from langchain.llms import OpenAI
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.chains import RetrievalQA

def generate_response(classifications_files_dict, openai_api_key, query_text, user_claims):
    # Load document if file is uploaded
    retrievers = []
    for classification, file in classifications_files_dict.items():
        if authorized(user_claims, classification):
            retrievers.append(build_retriever(file, openai_api_key))

    if len(retrievers) == 0:
        return "No files authorized for current user"
        
    weights = [1/len(retrievers)] * len(retrievers)
    # initialize the ensemble retriever
    ensemble_retriever = EnsembleRetriever(
        retrievers=retreivers, weights=weights
    )
    # Create QA chain
                          
    qa = RetrievalQA.from_chain_type(llm=OpenAI(openai_api_key=openai_api_key), chain_type='stuff', retriever=ensemble_retriever)
    return qa.run(query_text)

def authorized(claims, classification):
    # todo (dramdass)
    for claim in claims:
        if claim == classification:
            return True
    return False

def build_retriever(uploaded_file, openai_api_key):
    if uploaded_file is not None:
        documents = [uploaded_file.read().decode()]
    # Split documents into chunks
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    texts = text_splitter.create_documents(documents)
    # Select embeddings
    embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
    # Create a vectorstore from documents
    db = Chroma.from_documents(texts, embeddings)
    # Create retriever interface
    retriever = db.as_retriever()
    return retriever

# Page title
st.set_page_config(page_title='🦜🔗 Ask the Doc App')
st.title('🦜🔗 RBAC - Ask the Doc App')

# File upload
internal_file = st.file_uploader('Upload an internal document', type='pdf')
restricted_file = st.file_uploader('Upload a restricted document', type='pdf')
confidential_file = st.file_uploader('Upload a confidential document', type='pdf')

uploaded_files = {
    'internal': internal_file,
    'restricted': restricted_file,
    'confidential': confidential_file
}
disabled = (len(uploaded_files) == 0)

user_claims_string = st.text_input('Enter your claims separated by commas:', placeholder = 'Please provide a short summary.', disabled=disabled)
user_claims = user_claims_string.split(',')

# Query text
query_text = st.text_input('Enter your question:', placeholder = 'Please provide a short summary.', disabled=disabled)

disabled = disabled and not (user_claims and query_text)
# Form input and query
result = []
with st.form('myform', clear_on_submit=True):
    openai_api_key = st.text_input('OpenAI API Key', type='password', disabled=disabled)
    submitted = st.form_submit_button('Submit', disabled=disabled)
    if submitted and openai_api_key.startswith('sk-'):
        with st.spinner('Calculating...'):
            response = generate_response(uploaded_files, openai_api_key, query_text, user_claims)
            result.append(response)
            del openai_api_key

if len(result):
    st.info(response)
