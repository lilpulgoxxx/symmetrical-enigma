import os
import subprocess
import asyncio
from typing import Any, Dict
from fastapi import FastAPI, HTTPException, Response
from pydantic import BaseModel
import uvicorn
from langchain.prompts import PromptTemplate
from langchain_community.chat_models import ChatOllama
from langchain_core.output_parsers import JsonOutputParser
from langgraph.graph import END, StateGraph
from typing_extensions import TypedDict
import multiprocessing
from datetime import datetime
import time
from fastapi.responses import StreamingResponse, JSONResponse
import httpx
from functools import lru_cache

os.system("ollama serve &")

def download_ollama_model(model_name='hf.co/MaziyarPanahi/Llama-3.2-3B-Instruct-uncensored-GGUF:IQ1_S'):
    try:
        print(f"Descargando el modelo: {model_name}")
        subprocess.run(["ollama", "pull", model_name], check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error al descargar el modelo: {e}")
        raise

download_ollama_model("hf.co/MaziyarPanahi/Llama-3.2-3B-Instruct-uncensored-GGUF:IQ1_S")

app = FastAPI()

class QueryRequest(BaseModel):
    query: str

local_llm = 'hf.co/MaziyarPanahi/Llama-3.2-3B-Instruct-uncensored-GGUF:IQ1_S'
llama3 = ChatOllama(model=local_llm, stream=True)


@lru_cache(maxsize=1)
def get_global_data():
    return {
        'tokensxx': {
            'bos': '<|begin_of_text|>',
            'eot': '<|eot_id|>'
        }
    }

global_data = get_global_data()

cutting_knowledge_date = datetime.now()

def create_generate_prompt():
    today = datetime.now()
    template = f"""
        {global_data['tokensxx']['bos']}
        <|start_header_id|>system<|end_header_id|>
        Cutting Knowledge Date: {cutting_knowledge_date.strftime('%B %d %Y')}
        Today Date: {today.strftime('%d %b %Y')}
        You are an AI assistant named Cyrah for Research Question Tasks, that synthesizes web search results.
        Strictly use the following pieces of web search context to answer the question. If you don't know the answer, just say that you don't know.
        Keep the answer concise, but provide all of the details you can in the form of a research report.
        Only make direct references to material if provided in the context.
        {global_data['tokensxx']['eot']}
        <|start_header_id|>user<|end_header_id|>
        Question: {{question}}
        Web Search Context: {{context}}
        Answer:
        {global_data['tokensxx']['eot']}
        <|start_header_id|>assistant<|end_header_id|>"""
    return PromptTemplate(template=template, input_variables=["question", "context"])

generate_prompt = create_generate_prompt()


router_prompt = PromptTemplate(
    template=f"""
        {global_data['tokensxx']['bos']}
        <|start_header_id|>system<|end_header_id|>
        You are an expert at routing a user question to either the generation stage or web search.
        Use the web search for questions that require more context for a better answer, or recent events.
        Otherwise, you can skip and go straight to the generation phase to respond.
        You do not need to be stringent with the keywords in the question related to these topics.
        Give a binary choice 'web_search' or 'generate' based on the question.
        Return the JSON with a single key 'choice' with no preamble or explanation.
        Question to route: {{question}}
        {global_data['tokensxx']['eot']}
        <|start_header_id|>assistant<|end_header_id|>""",
    input_variables=["question"],
)

question_router = router_prompt | llama3 | JsonOutputParser()

query_prompt = PromptTemplate(
    template=f"""
        {global_data['tokensxx']['bos']}
        <|start_header_id|>system<|end_header_id|>
        You are an expert at crafting web search queries for research questions.
        More often than not, a user will ask a basic question that they wish to learn more about, however it might not be in the best format.
        Reword their query to be the most effective web search string possible.
        Return the JSON with a single key 'query' with no preamble or explanation.
        Question to transform: {{question}}
        {global_data['tokensxx']['eot']}
        <|start_header_id|>assistant<|end_header_id|>""",
    input_variables=["question"],
)

query_chain = query_prompt | llama3 | JsonOutputParser()

class GraphState(TypedDict):
    question: str
    generation: str
    search_query: str
    context: str

async def generate_stream(state):
    question = state["question"]
    context = state["context"]

    async def stream_response_generator():
         generate_chain = generate_prompt | llama3
         async for token in generate_chain.astream({"context": context, "question": question}):
            yield token.content
            
    return StreamingResponse(stream_response_generator(), media_type="text/plain")

def transform_query(state):
    print("Step: Optimizing Query for Web Search")
    question = state['question']
    gen_query = query_chain.invoke({"question": question})
    search_query = gen_query.get("query", "")
    return {"search_query": search_query}

def web_search(state):
    search_query = state['search_query']
    print(f'Step: Searching the Web for: "{search_query}"')
    return {"context": ""}

def route_question(state):
    print("Step: Routing Query")
    question = state['question']
    output = question_router.invoke({"question": question})
    if output.get('choice') == "web_search":
        print("Step: Routing Query to Web Search")
        return "websearch"
    elif output.get('choice') == 'generate':
        print("Step: Routing Query to Generation")
        return "generate"
    
    return "generate"


workflow = StateGraph(GraphState)
workflow.add_node("websearch", web_search)
workflow.add_node("transform_query", transform_query)
workflow.add_node("generate", generate_stream)

workflow.set_conditional_entry_point(
    route_question,
    {
        "websearch": "transform_query",
        "generate": "generate",
    },
)
workflow.add_edge("transform_query", "websearch")
workflow.add_edge("websearch", "generate")
workflow.add_edge("generate", END)

local_agent = workflow.compile()

async def run_agent(query):
    output = await local_agent.ainvoke({"question": query})
    
    if isinstance(output, StreamingResponse):
        return output
    elif "generation" not in output:
        print("Web search failed, using Llama model directly.")
        async def stream_response_generator():
            generate_chain = generate_prompt | llama3
            async for token in generate_chain.astream({"context": "", "question": query}):
                yield token.content
        return StreamingResponse(stream_response_generator(), media_type="text/plain")
    
    return output
    
app = FastAPI()


def run_uvicorn():
    uvicorn.run(app, host="0.0.0.0", port=7860, log_level="info")

@app.post("/query")
async def query_handler(request: QueryRequest):
    try:
        query = request.query
        result = await run_agent(query)
        
        if isinstance(result, StreamingResponse):
           return result
        elif "generation" in result:
            return JSONResponse(content={"generation": result["generation"]})
        else:
            return JSONResponse(content={"error": "Unexpected response format from agent."}, status_code=500)

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error en el procesamiento: {str(e)}")

@app.on_event("startup")
async def startup_event():
    print("Pre-loading Model...")
    try:
       llama3.invoke("hello")
       print("Model pre-loaded.")
    except Exception as e:
        print(f"Error pre-loading the model: {e}")

if __name__ == "__main__":
    uvicorn_process = multiprocessing.Process(target=run_uvicorn)
    uvicorn_process.start()
    uvicorn_process.join()
