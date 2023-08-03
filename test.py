from dotenv import load_dotenv
import streamlit as st
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.llms import OpenAI
from langchain.callbacks import get_openai_callback
import pandas as pd
import chardet
import io

def main():
  load_dotenv()

  # locale.setlocale(locale.LC_ALL, 'zh_CN')
  st.set_page_config(page_title="CSV Knowledge base")
  st.header("CSV knowledge base")
  # Upload files
  csv_file = st.file_uploader("Upload csv", type="csv")
  # Extract texts
  if csv_file is not None:
    rawdata = csv_file.read()
    result = chardet.detect(rawdata)
    encoding = result['encoding']
    s = io.StringIO(rawdata.decode(encoding,'ignore'))
    df = pd.read_csv(s)
    text = " ".join(df.astype(str).values.flatten())  # Assuming the CSV file contains text, join all the text together

    # Text splitter
    text_splitter = CharacterTextSplitter(
      separator="\n",
      chunk_size=1000,
      chunk_overlap=50,
      length_function=len
    )
    chunks = text_splitter.split_text(text)
    
    # Create embeddings
    embeddings = OpenAIEmbeddings()
    knowledge_base = FAISS.from_texts(chunks, embeddings)
    user_question = st.text_input("Ask me a question")
    if user_question:
      docs = knowledge_base.similarity_search(user_question)
      
      llm = OpenAI()
      chain = load_qa_chain(llm, chain_type="stuff")
      with get_openai_callback() as cb:
        response = chain.run(input_documents=docs, question=user_question)
        print(cb)
       
      # Write estimate tokens and costs   
      st.write(response)

if __name__ == '__main__':
    main()
