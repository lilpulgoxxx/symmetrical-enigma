import os
import subprocess
from typing import Any, Dict
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn
from langchain.prompts import PromptTemplate
from langchain_community.chat_models import ChatOllama
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_community.utilities import DuckDuckGoSearchAPIWrapper
from langchain_core.output_parsers import JsonOutputParser, StrOutputParser
from langgraph.graph import END, StateGraph
from typing_extensions import TypedDict
import multiprocessing
from datetime import datetime
import time
from fastapi.responses import StreamingResponse
import asyncio
import re

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
llama3 = ChatOllama(model=local_llm)

wrapper = DuckDuckGoSearchAPIWrapper(max_results=1)
web_search_tool = DuckDuckGoSearchRun(api_wrapper=wrapper)

global_data: Dict[str, Any] = {
    'models': {},
    'tokensxx': {
        'eos': '<|end_of-text|>',
        'pad': '<pad>',
        'unk': '<unk>',
        'bos': '<|begin_of_text|>',
        'sep': '<|sep|>',
        'cls': '<|cls|>',
        'mask': '<mask>',
        'eot': '<|eot_id|>',
        'eom': '<|eom_id|>',
        'lf': '<|0x0A|>'
    },
    'tokens': {
        'eos': 'eos_token',
        'pad': 'pad_token',
        'unk': 'unk_token',
        'bos': 'bos_token',
        'sep': 'sep_token',
        'cls': 'cls_token',
        'mask': 'mask_token'
    },
    'model_metadata': {},
    'eos': {},
    'pad': {},
    'padding': {},
    'unk': {},
    'bos': {},
    'sep': {},
    'cls': {},
    'mask': {},
    'eot': {},
    'eom': {},
    'lf': {},
    'max_tokens': {},
    'tokenizers': {},
    'model_params': {},
    'model_size': {},
    'model_ftype': {},
    'n_ctx_train': {},
    'n_embd': {},
    'n_layer': {},
    'n_head': {},
    'n_head_kv': {},
    'n_rot': {},
    'n_swa': {},
    'n_embd_head_k': {},
    'n_embd_head_v': {},
    'n_gqa': {},
    'n_embd_k_gqa': {},
    'n_embd_v_gqa': {},
    'f_norm_eps': {},
    'f_norm_rms_eps': {},
    'f_clamp_kqv': {},
    'f_max_alibi_bias': {},
    'f_logit_scale': {},
    'n_ff': {},
    'n_expert': {},
    'n_expert_used': {},
    'causal_attn': {},
    'pooling_type': {},
    'rope_type': {},
    'rope_scaling': {},
    'freq_base_train': {},
    'freq_scale_train': {},
    'n_ctx_orig_yarn': {},
    'rope_finetuned': {},
    'ssm_d_conv': {},
    'ssm_d_inner': {},
    'ssm_d_state': {},
    'ssm_dt_rank': {},
    'ssm_dt_b_c_rms': {},
    'vocab_type': {},
    'model_type': {},
    "general.architecture": {},
    "general.type": {},
    "general.name": {},
    "general.finetune": {},
    "general.basename": {},
    "general.size_label": {},
    "general.license": {},
    "general.license.link": {},
    "general.tags": {},
    "general.languages": {},
    "general.organization": {},
    "general.base_model.count": {},
    'general.file_type': {},
    "phi3.context_length": {},
    "phi3.rope.scaling.original_context_length": {},
    "phi3.embedding_length": {},
    "phi3.feed_forward_length": {},
    "phi3.block_count": {},
    "phi3.attention.head_count": {},
    "phi3.attention.head_count_kv": {},
    "phi3.attention.layer_norm_rms_epsilon": {},
    "phi3.rope.dimension_count": {},
    "phi3.rope.freq_base": {},
    "phi3.attention.sliding_window": {},
    "phi3.rope.scaling.attn_factor": {},
    "llama.block_count": {},
    "llama.context_length": {},
    "llama.embedding_length": {},
    "llama.feed_forward_length": {},
    "llama.attention.head_count": {},
    "llama.attention.head_count_kv": {},
    "llama.rope.freq_base": {},
    "llama.attention.layer_norm_rms_epsilon": {},
    "llama.attention.key_length": {},
    "llama.attention.value_length": {},
    "llama.vocab_size": {},
    "llama.rope.dimension_count": {},
    "deepseek2.block_count": {},
    "deepseek2.context_length": {},
    "deepseek2.embedding_length": {},
    "deepseek2.feed_forward_length": {},
    "deepseek2.attention.head_count": {},
    "deepseek2.attention.head_count_kv": {},
    "deepseek2.rope.freq_base": {},
    "deepseek2.attention.layer_norm_rms_epsilon": {},
    "deepseek2.expert_used_count": {},
    "deepseek2.leading_dense_block_count": {},
    "deepseek2.vocab_size": {},
    "deepseek2.attention.kv_lora_rank": {},
    "deepseek2.attention.key_length": {},
    "deepseek2.attention.value_length": {},
    "deepseek2.expert_feed_forward_length": {},
    "deepseek2.expert_count": {},
    "deepseek2.expert_shared_count": {},
    "deepseek2.expert_weights_scale": {},
    "deepseek2.rope.dimension_count": {},
    "deepseek2.rope.scaling.type": {},
    "deepseek2.rope.scaling.factor": {},
    "deepseek2.rope.scaling.yarn_log_multiplier": {},
    "qwen2.block_count": {},
    "qwen2.context_length": {},
    "qwen2.embedding_length": {},
    "qwen2.feed_forward_length": {},
    "qwen2.attention.head_count": {},
    "qwen2.attention.head_count_kv": {},
    "qwen2.rope.freq_base": {},
    "qwen2.attention.layer_norm_rms_epsilon": {},
    "general.version": {},
    "general.datasets": {},
    "tokenizer.ggml.model": {},
    "tokenizer.ggml.pre": {},
    "tokenizer.ggml.tokens": {},
    "tokenizer.ggml.token_type": {},
    "tokenizer.ggml.merges": {},
    "tokenizer.ggml.bos_token_id": {},
    "tokenizer.ggml.eos_token_id": {},
    "tokenizer.ggml.unknown_token_id": {},
    "tokenizer.ggml.padding_token_id": {},
    "tokenizer.ggml.add_bos_token": {},
    "tokenizer.ggml.add_eos_token": {},
    "tokenizer.ggml.add_space_prefix": {},
    "tokenizer.chat_template": {},
    "quantize.imatrix.file": {},
    "quantize.imatrix.dataset": {},
    "quantize.imatrix.entries_count": {},
    "quantize.imatrix.chunks_count": {},
    "general.quantization_version": {},
    'n_lora_q': {},
    'n_lora_kv': {},
    'n_expert_shared': {},
    'n_ff_exp': {},
    "n_layer_dense_lead": {},
    "expert_weights_scale": {},
    "rope_yarn_log_mul": {},
    'request_count': {},
    'last_request_time': {},
    'active_requests': {},
    'model_load_status': {},
    'model_error_logs': {},
    'average_response_time':{},
    'tokens_processed':{},
    'gpu_utilization': {},
    'request_queue_size': {},
    'model_peak_memory': {},
    'model_cumulative_load_time': {},
    'model_inference_count':{},
    'model_token_generation_speed': {},
    'model_hardware_info':{},
    'request_latency_per_token': {},
    'total_running_time': {},
    'system_load': {},
    'system_start_time': time.time(),
    'model_last_successful_inference': {},
    'model_performance_history':{},
    'request_history': {},
    'model_load_start_time': {},
    'model_last_access_time':{},
    'model_total_inference_time':{},
    'model_average_token_latency':{},
    'model_max_memory_usage':{},
    'model_min_memory_usage': {},
    'model_current_memory_usage':{},
    'model_parameters_count':{},
    'model_max_tokens_limit':{},
    'model_context_window_size': {},
    'model_tensor_dtype': {},
    'model_device_type':{},
    'model_quantization_type':{},
    'model_architecture_name':{},
    'model_license_name':{},
    'model_base_name': {},
    'model_tags_list':{},
    'model_languages_list':{},
    'model_datasets_list':{},
    'model_file_type':{},
    'model_version':{},
    'model_organization_name':{},
    'model_finetuned_status':{},
    'model_vocabulary_size':{},
    'tokenizer_type': {},
    'tokenizer_merges_count':{},
    'tokenizer_tokens_count':{},
    'tokenizer_add_bos_token_status':{},
    'tokenizer_add_eos_token_status':{},
    'tokenizer_add_prefix_token_status':{},
    'tokenizer_unknown_token_id':{},
    'tokenizer_padding_token_id':{},
    'tokenizer_chat_template_name':{},
    'model_sliding_window_size':{},
    'model_rope_dimension_size': {},
    'model_rope_scaling_type':{},
    'model_rope_scaling_factor':{},
    'model_rope_base_freq':{},
    'model_feed_forward_size':{},
    'model_embedding_size':{},
    'model_head_count':{},
    'model_kv_head_count':{},
    'model_layer_norm_eps': {},
    'model_attention_layer_norm_eps': {},
    'model_block_count':{},
    'model_attention_key_length':{},
    'model_attention_value_length':{},
    'model_expert_count':{},
    'model_expert_used_count':{},
    'model_quantization_version':{},
    'model_lora_rank':{},
    'system_cpu_load_history':{},
    'system_ram_load_history':{},
    'model_load_times_history':{},
    'model_generation_speed_history':{},
    'request_error_count':{},
    'model_parameter_count_history': {},
    'model_layer_names':{},
    'model_optimizer_type':{},
    'model_loss_function_type':{},
    'model_gradient_clipping_value':{},
    'model_learning_rate':{},
    'model_weight_decay_value':{},
    'model_bias_values': {},
    'model_dropout_values':{},
    'model_training_batch_size':{},
    'model_training_sequence_length':{},
    'model_training_data_size': {},
    'model_training_steps': {},
    'model_warmup_steps':{},
    'model_gradient_accumulation_steps': {},
    'model_training_epoch_count':{},
    'model_activation_functions':{},
    'model_embedding_layer_type': {},
    'model_attention_mechanism_type':{},
    'model_normalization_layer_type': {},
    'model_pooling_layer_type':{},
    'model_rope_implementation_type':{},
    'model_pos_encoding_type': {},
    'model_number_of_attention_heads_per_layer':{},
    'model_number_of_feed_forward_layers': {},
    'model_number_of_decoder_layers':{},
    'model_number_of_encoder_layers': {},
    'model_cross_attention_layers_count':{},
    'model_is_encoder_decoder':{},
    'model_head_size':{},
    'model_key_size':{},
    'model_value_size':{},
    'model_query_size':{},
    'model_kv_cache_type': {},
    'model_is_bidirectional_attention':{},
    'model_attention_dropout':{},
    'model_is_moe':{},
    'model_num_moe_experts_used': {},
    'model_expert_selection_mechanism':{},
    'model_expert_capacity_factor':{},
    'model_moe_gate_layer_type':{},
    'model_moe_top_k_value':{},
    'model_quantization_bits':{},
    'model_quantization_scheme':{},
    'model_weight_quantization_method':{},
    'model_activation_quantization_method':{},
    'model_bias_quantization_method':{},
    'model_use_flash_attention':{},
    'model_use_xformers_attention':{},
    'model_sequence_parallelism_degree':{},
    'model_tensor_parallelism_degree':{},
    'model_data_parallelism_degree':{},
    'model_pipeline_parallelism_degree': {},
    'model_sharding_strategy': {},
    'model_mixed_precision_type':{},
    'model_gradient_checkpointing_status':{},
    'model_is_cpu_offload_enabled':{},
    'model_optimizer_beta1':{},
    'model_optimizer_beta2':{},
    'model_optimizer_epsilon': {},
    'model_gradient_norm_clipping_value':{},
    'model_weight_initialization_scheme':{},
    'model_parameter_sharing_strategy': {},
    'model_parameter_tying_strategy': {},
    'model_custom_kernels_used': {},
    'model_is_activation_checkpointing':{},
    'model_is_gradient_accumulation':{},
    'model_use_layer_norm_before_attention':{},
    'model_use_layer_norm_after_attention':{},
    'model_use_layer_norm_before_ffn':{},
    'model_use_layer_norm_after_ffn':{},
    'model_use_bias_in_qkv_projection':{},
    'model_use_bias_in_output_projection':{},
    'model_use_bias_in_ffn':{},
    'model_use_bias_in_attention_gate':{},
    'model_norm_eps_value': {},
    'model_norm_type': {},
    'model_is_embedding_shared': {},
    'model_is_output_layer_shared':{},
    'eval': {},
    'time': {},
    'token': {},
    'tokens': {},
    'pads': {},
    'model': {},
    'base': {},
    'model_base': {},
    'perhaps': {},
    'word': {},
    'words': {},
    'start': {},
    'stop': {},
    'run': {},
    'runs': {},
    'ms': {},
    'vocabulary': {},
    'timeout': {},
    'load': {},
    'load_time': {},
    'bas': {},
    'tok': {},
    'second': {},
    'seconds': {},
    'graph': {},
    'load_model': {},
    'end': {},
    'llama_perf_context_print': {},
    'llm_load_print_meta': {},
    'model_type': {},
    'image_model': {}
}


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
        Return the JSON with a single key 'choice' with no premable or explanation.
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
        Return the JSON with a single key 'query' with no premable or explanation.
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
        generate_chain = generate_prompt | llama3 | StrOutputParser()
        async for token in generate_chain.astream({"context": context, "question": question}):
             yield token

    return StreamingResponse(stream_response_generator(), media_type="text/plain")

def transform_query(state):
    print("Step: Optimizing Query for Web Search")
    question = state['question']
    gen_query = query_chain.invoke({"question": question})
    search_query = gen_query.get("query", "")
    return {"search_query": search_query}

async def web_search(state):
    search_query = state['search_query']
    print(f'Step: Searching the Web for: "{search_query}"')
    try:
        search_result = web_search_tool.invoke(search_query)
        if isinstance(search_result, str):
            print(f"Respuesta de búsqueda web es cadena: {search_result}")
            return {"context": search_result}
        elif isinstance(search_result, dict):
            return {"context": search_result}
        else:
            raise ValueError("Respuesta de búsqueda web no es válida")
    except Exception as e:
        print(f"Web search failed: {e}")

        async def stream_response_generator():
            generate_chain = generate_prompt | llama3 | StrOutputParser()
            async for token in generate_chain.astream({"context": "", "question": state['question']}):
              yield token

        return {"context":  StreamingResponse(stream_response_generator(), media_type="text/plain")}


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

async def run_agent_parallel(query):
    output = await local_agent.ainvoke({"question": query})
    if isinstance(output, StreamingResponse):
         return output
    elif "generation" not in output:
        print("Web search failed, using Ollama model directly.")

        async def stream_response_generator():
            generate_chain = generate_prompt | llama3 | StrOutputParser()
            async for token in generate_chain.astream({"context": "", "question": query}):
              yield token
        return StreamingResponse(stream_response_generator(), media_type="text/plain")

    return output.get("generation", "")


async def process_query_in_parallel(query):
     with multiprocessing.Pool(processes=multiprocessing.cpu_count()) as pool:
         tasks = [run_agent_parallel(query) for _ in range(multiprocessing.cpu_count())]
         results = await asyncio.gather(*tasks)
     return results[0]

@app.post("/query")
async def query_handler(request: QueryRequest):
    try:
        query = request.query
        result = await process_query_in_parallel(query)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error en el procesamiento: {str(e)}")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=7860)
