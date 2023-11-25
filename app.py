from flask import Flask, request, jsonify
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from torch import cuda
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
import pinecone
import time
import pandas as pd
from transformers import AutoModelForCausalLM, AutoConfig
from transformers.file_utils import is_torch_available
from transformers import BitsAndBytesConfig
import transformers
from langchain.vectorstores import Pinecone
from langchain.llms import HuggingFacePipeline
from langchain.chains import RetrievalQA

app = Flask(__name__)
def function(input):
    embed_model_id = 'sentence-transformers/all-MiniLM-L6-v2'
    device = f'cuda:{cuda.current_device()}' if cuda.is_available() else 'cpu'
    embed_model = HuggingFaceEmbeddings(
        model_name=embed_model_id,
        model_kwargs={'device': device},
        encode_kwargs={'device': device, 'batch_size': 32}
    )
    pinecone.init(
        api_key='c2c69825-f57f-4dbc-a7fb-467505a10990',
        environment='gcp-starter'
    )
    index_name = 'chaabii'

    index = pinecone.Index('chaabii')
    if index_name not in pinecone.list_indexes():
        pinecone.create_index(
            index_name,
            dimension=len(embeddings[0]),
            metric='cosine'
        )
        while not pinecone.describe_index(index_name).status['ready']:
            time.sleep(1)
    index = pinecone.Index(index_name)
    index.describe_index_stats()

    data = pd.read_csv("bigBasketProducts.csv")
    data = data.dropna(subset=['description'])
    data = data.dropna(subset=['product'])
    data = data.dropna(subset=['rating'])

    if is_torch_available():
        from torch import cuda, bfloat16

    model_id = '01-ai/Yi-6B'
    if is_torch_available() and cuda.is_available():
        device = f'cuda:{cuda.current_device()}'
    else:
        device = 'cpu'

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type='nf4',
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=bfloat16
    )

    model_config = AutoConfig.from_pretrained(
        model_id,
        torch_dtype=bfloat16 if is_torch_available() else None
    )

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        config=model_config,
        quantization_config=bnb_config if is_torch_available() else None
    )
    model.eval()

    tokenizer = transformers.AutoTokenizer.from_pretrained(model_id)
    generate_text = transformers.pipeline(
        model=model, tokenizer=tokenizer,
        return_full_text=True,
        task='text-generation',
        temperature=0.0,
        max_new_tokens=512,
        repetition_penalty=1.1
    )

    llm = HuggingFacePipeline(pipeline=generate_text)
    text_field = 'text'
    vectorstore = Pinecone(
        index, embed_model.embed_query, text_field
    )

    rag_pipeline = RetrievalQA.from_chain_type(
        llm=llm, chain_type='stuff',
        retriever=vectorstore.as_retriever()
    )
    return rag_pipeline(input)['result']

@app.route('/generate', methods=['POST'])
def generate_text():
    try:
        input_text = request.form['input_text']
        output = function(input_text)
        return jsonify({"output": input_text})

    except Exception as e:
        return jsonify({"error": str(e)})

if __name__ == '__main__':
    app.run(port=5000)
