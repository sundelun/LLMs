# LLMs
#Inside terminal: 
#name of folder, can be any name
1. mkdir llms
2. cd llms
#create virtual environment, second venv is the name of folder, can be any name
3. python3 -m venv venv
# Activate virtual environment, should activate before running.
4. source venv/bin/activate
# install packages
5. pip install langchain pdfplumber python-dotenv streamlit faiss-cpu openai tiktoken
6. inside venv(name) folder, create a .env file with OPENAI_API_KEY='your_key'
7. download app.py and test.py
8. When using pdfs, in terminal type in "streamlit run app.py" and when using csv, in terminal type in "streamlit run test.py"
