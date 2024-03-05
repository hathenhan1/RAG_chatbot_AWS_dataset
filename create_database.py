from langchain_community.document_loaders import DirectoryLoader 
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from langchain_community.embeddings import OpenAIEmbeddings
from langchain.vectorstores.chroma import Chroma
from dotenv import load_dotenv
import os
import shutil
'''
   os and shutil library provides function for manipulating the operating system,
   including operations with files and directories
'''

CHROMA_PATH = "chroma" 
DATA_PATH = "data/books" 

#main function
def main():
    load_dotenv(".env")
    generate_data_store()



def generate_data_store():
    documents = load_document() 
    chunks = split_text(documents)
    save_to_chroma(chunks)
    
    
#Load_document function
def load_document():
    '''
    Loader:
        DirectoryLoader is a class in library.
        DATA_PATH and glob are two parameters which passed to DirectoryLoader
            _"DATA_PATH" is the directory to the dataset "alice_in_wonder_land.md"
            _"glop=*.md" allows which file only have .md tag to be loaded 
    '''
    loader = DirectoryLoader(DATA_PATH, glob="*.md") 
    #Call the method load() in Loader to find the folder  
    documnets = loader.load()
    
    return documnets

#Split_text function
def split_text(documents: list[Document]):
    '''
        RecursiveCharacterTextSplitter: is a class of Langchain library
        "This class is used to split documents into text segments based on character count"
        {
             chunk_size: Size of each text segments(about 300 characters)
             chunk_overlap: The number of overlapping characters between consecutive paragraphs of text
             length_function = len : is the function to measure the length of text
             add_start_index = True 
             ---------------------------------------------------------------------------------
             text_splitter.split_documents() Call the split function to split documents
        }
    '''
    text_splitter = RecursiveCharacterTextSplitter( 
        chunk_size = 300, 
        chunk_overlap = 100,
        length_function=len,
        add_start_index = True,
    )
    chunks = text_splitter.split_documents(documents)

    #print length of documents and length of chunks when split documents to chunks 
    print(f"Split {len(documents)} documents into {len(chunks)} chunks")

    document = chunks[10] #documents have value equal 10th chunk 

    print(document.page_content) #print content in document (10th chunk)

    print(document.metadata)#print the directory to which file of data we choose and the started index of chunk we choose

    return chunks 

def save_to_chroma(chunks: list[Document]):
    # The os.path.exists check the folder have already existed at CHROMA_PATH path. 
    #If chroma db exist, the shutil will be remove everthing in folder 
    if os.path.exists(CHROMA_PATH):
        shutil.rmtree(CHROMA_PATH)
    
    #Create a new DB from documents with chunks, OpenAIEmbeddings and the presist_directory=CHROMA_PATH
    db = Chroma.from_documents(
        chunks , OpenAIEmbeddings(), persist_directory=CHROMA_PATH,
    )

    #Save Chroma database to disk
    db.persist
    print(f"Saved {len(chunks)} chunks to {CHROMA_PATH}")

if __name__ == "__main__":
    main()
