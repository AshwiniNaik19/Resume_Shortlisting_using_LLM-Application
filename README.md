uvicorn test:app --port 8000
$env:NO_PROXY="localhost,127.0.0.1,https://prabhatopenaiservice.openai.azure.com/"

python app.py
$env:NO_PROXY="127.0.0.1,localhost"
