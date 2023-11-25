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

def function(input):
    embed_model_id = 'sentence-transformers/all-MiniLM-L6-v2'
    device = f'cuda:{cuda.current_device()}' if cuda.is_available() else 'cpu'
    
    '''Initialising the Embedding pipeline for transformation into Vector Embeddings'''

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

    '''Function for creating Vector Embeddings and storing them in Pinecone Vector Index.'''

    batch_size = 32
    for i in range(0, len(data), batch_size):
        i_end = min(len(data), i+batch_size)
        batch = data.iloc[i:i_end]
        ids = [f"{x['index']}" for i, x in batch.iterrows()]
        texts = [x['description'] for i, x in batch.iterrows()]
        embeds = embed_model.embed_documents(texts)
        metadata = [
            {'text': x['description'],
             'product' : x['product'],
             'category': x['category'],
             'sub_category' : x['sub_category'],
             'brand': x['brand'],
             'sale_price' : x['sale_price'],
             'market_price' : x['market_price'],
             'type' : x['type'],
             'rating' : x['rating'],
             } for i, x in batch.iterrows()
        ]
        index.upsert(vectors=zip(ids, embeds, metadata))

    if is_torch_available():
        from torch import cuda, bfloat16

    '''Initializing the LLM model which we will use'''
    model_id = '01-ai/Yi-6B'
    if is_torch_available() and cuda.is_available():
        device = f'cuda:{cuda.current_device()}'
    else:
        device = 'cpu'

    '''set quantization configuration to load large model with less GPU memory'''
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
