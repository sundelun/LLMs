from dotenv import load_dotenv
import streamlit as st
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.llms import OpenAI
from langchain.callbacks import get_openai_callback
import pdfplumber
import pickle
import faiss

def main():
  load_dotenv()

  st.set_page_config(page_title="PDF knowledge base")
  st.header("PDF KNOWLEDGE BASE")
  #Upload files
  pdf = st.file_uploader("UPLOAD PDF", type="pdf")
  # extract text
  if pdf is not None:
    text=""
    with pdfplumber.open(pdf) as pdf_reader:
      for page in pdf_reader.pages:
        text += page.extract_text()
    # text splitter
    text_splitter = CharacterTextSplitter(
      separator="\n",
      chunk_size=1000,
      chunk_overlap=50,
      length_function=len
    )
    chunks = text_splitter.split_text(text)
    
    # create embeddings
    embeddings = OpenAIEmbeddings(openai_api_version='2020-11-07')
    knowledge_base = FAISS.from_texts(chunks, embeddings)
    faiss.write_index(knowledge_base.index,'knowledge_base.index')
    
    # Write FAISS index and all text data, comment till f.write if no needed.
    with open('chunks.pkl','wb') as f:
      pickle.dump(chunks,f)
    with open('text_data.txt', 'w', encoding='utf-8') as f:
      for chunk in chunks:
        f.write(chunk + '\n')
        
    # Sum of all tokens
    knowledge_base_token_count = sum(len(chunk.split()) for chunk in chunks)
    print(f"Knowledge Base Token Count: {knowledge_base_token_count}")
    user_question = st.text_input("Ask me a question: ")
    if user_question:
      docs = knowledge_base.similarity_search(user_question)
      
      llm = OpenAI()
      chain = load_qa_chain(llm, chain_type="stuff")
      with get_openai_callback() as cb:
        response = chain.run(input_documents=docs, question=user_question)
        print(cb)
      
      # Print the estimate costs and tokens for each response
      st.write(response)

if __name__ == '__main__':
    main()
