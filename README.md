NTRODUCTION
This Medical Chatbot uses the latest MoE model Qwen2 1.5B model and levrages LORA and RAG to deliver results far better than the base model. We have used the Gale Encyclopedia of Medicine through Qudrant vector database to levrage RAG and add "memory" to our model. The model can be downloaded from here "Irathernotsay/qwen2-1.5B-medical_qa-Finetune" or "https://huggingface.co/Irathernotsay/qwen2-1.5B-medical_qa-Finetune-Q4_K_M-GGUF" if you want inference on CPU. We have used 4 bit quantization to be able to run the model on CPU using the llama.cpp.

SETUP
Clone the project from github and create a virtual environment using conda in python=3.9 conda create -n myenv python=3.9 Move to the project directory using cd and install all the project requirements using your anaconda prompt pip install -r requirement.txt --use-deprecated=legacy-resolver Use docker to initialize the vector database. The details can be founded here. Run the ingest.py file to fill the vector database with the data. You can add any amount of data you want for vector database. Run the app.py file to run the chatbot web application locally. The app is made with gradio.

TODO
• Select the appropriate model. • Finetune the model on https://huggingface.co/datasets/ruslanmv/ai-medical-chatbot dataset. • Integrate RAG with the chatbot. • Build a website for the chatbot • Improve the UI • Deploy the model
