from langchain.document_loaders import PyPDFLoader, OnlinePDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Pinecone
from sentence_transformers import SentenceTransformer
from langchain.chains.question_answering import load_qa_chain
import pinecone
import os
import base64
import streamlit as st
from langchain.llms import LlamaCpp
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from huggingface_hub import hf_hub_download
from langchain.chains.question_answering import load_qa_chain

os.environ["HUGGINGFACEHUB_API_TOKEN"] = '' #put your hugging face api token here
os.environ["PINECONE_API_KEY"]= ''#replace this with your pinecone api key, find it in your pinecone account
os.environ["PINECONE_API_ENV"]="us-west1-gcp-free"#you will find the pinecone api env in your pinecone account details
PINECONE_API_KEY = ''#replace this with your pinecone api key, find it in your pinecone account
PINECONE_API_ENV = "us-west1-gcp-free"#you will find the pinecone api env in your pinecone account details
print("loading data")
#load data
loader = PyPDFLoader("sample_election.pdf")
data = loader.load()
print("done")
print("splitting documents")
#split document
text_splitter=RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
docs=text_splitter.split_documents(data)
print("done")
print("creating embeddings")
#create embeddings
embeddings=HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')

print("done")
print("initializing pinecone")
# initialize pinecone
pinecone.init(
    api_key=PINECONE_API_KEY,  # find at app.pinecone.io
    environment=PINECONE_API_ENV  # next to api key in console
)
index_name = 'llama2-ele' # put in the name of your pinecone index here
docsearch = Pinecone.from_existing_index(index_name, embeddings)
#docsearch=Pinecone.from_texts([t.page_content for t in docs], embeddings, index_name=index_name)
print("done")

print("initializing llama2")
callback_manager = CallbackManager([StreamingStdOutCallbackHandler()])
model_name_or_path = "TheBloke/Llama-2-13B-chat-GGML"
model_basename = "llama-2-13b-chat.ggmlv3.q5_1.bin"
model_path = '/Users/admin/.cache/huggingface/hub/models--TheBloke--Llama-2-13B-chat-GGML/snapshots/47d28ef5de4f3de523c421f325a2e4e039035bab/llama-2-13b-chat.ggmlv3.q5_1.bin'

n_gpu_layers = 40  # Change this value based on your model and your GPU VRAM pool.
n_batch = 256  # Should be between 1 and n_ctx, consider the amount of VRAM in your GPU.

# Loading model,
llm = LlamaCpp(
    model_path=model_path,
    max_tokens=256,
    n_gpu_layers=n_gpu_layers,
    n_batch=n_batch,
    callback_manager=callback_manager,
    n_ctx=2048,
    verbose=False,
)
print("done")
print("creating chain")
chain=load_qa_chain(llm, chain_type="stuff")
print("done")


def get_base64_of_bin_file(bin_file):
    with open(bin_file, 'rb') as f:
        data = f.read()
    return base64.b64encode(data).decode()

def set_png_as_page_bg(png_file):
    bin_str = get_base64_of_bin_file(png_file)
    page_bg_img = '''
    <style>
    body {
    background-image: url("data:image/png;base64,%s");
    background-size: cover;
    }
    </style>
    ''' % bin_str
    
    st.markdown(page_bg_img, unsafe_allow_html=True)
    return

set_png_as_page_bg('background.png')

st.header('Chat with your document using Llama2, langchain, Vector DB')
st.write('Hi There, How can I help you today?')
query = st.text_input('Search what you are looking for. ')
docs=docsearch.similarity_search(query)

if st.button('Chat with Doc- AI assistant'):
    #chain.run(input_documents=docs, question=query)
    st.write('Here are some matching parts from the documents store which you might find useful:  ')
    for doc in docs:
        st.markdown(doc)
    st.write('Here is what I found as per the documents:')
    st.write(chain.run(input_documents=docs, question=query))
