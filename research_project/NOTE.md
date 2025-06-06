docker run -v ./chroma-data:/data -p 8000:8000 chromadb/chroma

Connect to Chroma at: http://localhost:8000
Getting started guide: https://docs.trychroma.com/docs/overview/getting-started




cd examples/acr_demo
streamlit run app.py  --server.port 8501 --server.address localhost



export PYTHONPATH=$PWD