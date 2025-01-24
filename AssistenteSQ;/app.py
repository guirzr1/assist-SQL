from sqlalchemy import create_engine, MetaData

url = 'ecommerce.db'
engine = create_engine(f'sqlite:///{url}')

metadata_obj = MetaData()
metadata_obj.reflect(engine)

import os

key = os.getenv("GROQ_API")

modelo="llama-3.1-70b-versatile"
modelo_hf_emb="BAAI/bge-m3"

from llama_index.core import Settings
from llama_index.llms.groq import Groq
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

Settings.llm = Groq(model=modelo, api_key = key)
Settings.embed_model = HuggingFaceEmbedding(model_name = modelo_hf_emb)

from llama_index.core import SQLDatabase
from llama_index.core.objects import SQLTableNodeMapping

sql_database = SQLDatabase(engine)
table_node_map = SQLTableNodeMapping(sql_database)

llm = Groq(model=modelo, api_key = key)

def gerar_descricao_tabela(nome_tabela, df_amostra):
    prompt = f"""
    Analise a amostra da tabela '{nome_tabela}' abaixo e forneça uma curta e breve descrição do conteúdo dessa tabela.
    Informe até o máximo de 5 valores únicos de cada coluna.


    Amostra da Tabela:
    {df_amostra}

    Descrição:
    """

    resposta = llm.complete(prompt = prompt)

    return resposta.text

import pandas as pd

nomes_tabelas = metadata_obj.tables.keys()
dicionario_tabelas = {}

for nome_tabela in nomes_tabelas:
  df = pd.read_sql_table(nome_tabela,engine)
  df_amostra = df.head(5).to_string()

  descricao = gerar_descricao_tabela(nome_tabela, df_amostra)
  dicionario_tabelas[nome_tabela] = descricao
  print(f'Tabela: {nome_tabela}\n Descrição: {descricao}')
  print('-'*15)

from llama_index.core.objects import SQLTableSchema
from llama_index.core import VectorStoreIndex
from llama_index.core.objects import ObjectIndex

table_schema_objs = [
    SQLTableSchema(table_name= nome_tabela,context_str= dicionario_tabelas[nome_tabela])
    for nome_tabela in nomes_tabelas
]

obj_index = ObjectIndex.from_objects(table_schema_objs,table_node_map, VectorStoreIndex)
obj_retriever = obj_index.as_retriever(similarity_top_k=1)

texto2sql = """Dada uma pergunta em linguagem natural, crie uma consulta {dialect} sintaticamente correta para executar e, em seguida, verifique os resultados da consulta e retorne a resposta. Você pode ordenar os resultados por uma coluna relevante para retornar os exemplos mais informativos no banco de dados.

Nunca consulte todas as colunas de uma tabela específica. Pergunte apenas por algumas colunas relevantes, de acordo com a pergunta.

Preste atenção para usar apenas os nomes de colunas que você pode ver na descrição do esquema. Tenha cuidado para não consultar colunas que não existem. Preste atenção em qual coluna está em qual tabela. Além disso, qualifique os nomes das colunas com o nome da tabela quando necessário.

Use o seguinte formato, cada um em uma linha:

Pergunta: Pergunta aqui
ConsultaSQL: Consulta SQL para executar
ResultadoSQL: Resultado da ConsultaSQL
Resposta: Resposta final aqui

Use apenas as tabelas listadas abaixo.

{schema}

Pergunta: {pergunta_user}
ConsultaSQL:
"""

from llama_index.core import PromptTemplate

prompt_1 = PromptTemplate(texto2sql, dialect = engine.dialect.name)

from typing import List

def descricao_tabela (schema_tabelas: List[SQLTableSchema]):
  descricao_str = []
  for tabela_schema in schema_tabelas:
    info_tabela = sql_database.get_single_table_info(tabela_schema.table_name)
    info_tabela += (' A descrição da tabela é: '+tabela_schema.context_str)

    descricao_str.append(info_tabela)
  return '\n\n'.join(descricao_str)

from llama_index.core.query_pipeline import FnComponent

contexto_tabela = FnComponent(fn=descricao_tabela)

from llama_index.core.llms import ChatResponse

def resposta_sql (resposta: ChatResponse) -> str:
  conteudo_resposta = resposta.message.content

  sql_consulta = conteudo_resposta.split("ConsultaSQL: ", 1)[-1].split("ResultadoSQL: ", 1)[0]
  return sql_consulta.strip().strip('```').strip()

consulta_sql = FnComponent(fn=resposta_sql)

from llama_index.core.retrievers import SQLRetriever

resultado_sql = SQLRetriever(sql_database)

prompt_2_str = '''
    Você é o "Assitente de consulta de banco de dados da Zoop".
    Dada a seguinte pergunta, a consulta SQL correspondente e o resultado SQL, responda à pergunta de modo agradável e objetivamente.
    Evite iniciar conversas com cumprimentos e apresentações, como "Olá".

    Pergunta: {pergunta_user}
    Consulta SQL: {consulta}
    Resultado SQL: {resultado}
    Resposta:
    '''

prompt_2 = PromptTemplate(
    prompt_2_str,
)

from llama_index.core.query_pipeline import QueryPipeline, InputComponent

qp = QueryPipeline(
    modules = {
        'entrada': InputComponent(),
        'acesso_tabela': obj_retriever,
        'contexto_tabela': contexto_tabela,
        'prompt_1': prompt_1,
        'llm_1': llm,
        'consulta_sql': consulta_sql,
        'resultado_sql': resultado_sql,
        'prompt_2': prompt_2,
        'llm_2': llm,
    },
    verbose=False
)

qp.add_chain(['entrada', 'acesso_tabela', 'contexto_tabela'])
qp.add_link('entrada', 'prompt_1', dest_key='pergunta_user')
qp.add_link('contexto_tabela', 'prompt_1', dest_key='schema')
qp.add_chain(['prompt_1', 'llm_1', 'consulta_sql', 'resultado_sql'])
qp.add_link('entrada', 'prompt_2', dest_key='pergunta_user')
qp.add_link('consulta_sql', 'prompt_2', dest_key= 'consulta')
qp.add_link('resultado_sql', 'prompt_2', dest_key='resultado')
qp.add_link('prompt_2', 'llm_2')

def entrada_saida (msg_user: str):
  saida = qp.run(query=msg_user)
  return str(saida.message.content)

def adicao_historico(msg_user, historico):
  msg_assistente = entrada_saida(msg_user)

  historico.append([msg_user, msg_assistente])
  return msg_assistente, historico

import gradio as gr

with gr.Blocks() as demo:
    gr.Markdown('## Chat com Assistente SQL')
    gr.Markdown(
        '''
        Este é um assistente SQL interativo, projetado para responder perguntas sobre os dados da loja Zoop.
        Insira sua pergunta no campo abaixo e o assistente irá responder com base no resultado da consulta SQL
        nos dados disponíveis.
        '''
    )
    chatbot = gr.Chatbot(label='Chat com Assistente')
    msg = gr.Textbox(label='Digite a sua pergunta e tecle Enter para enviar',
                     placeholder='Digite o texto aqui.')
    limpeza = gr.Button('Limpar a conversa')
    def atualizar_historico (msg_user, historico):
      msg_assistente, historico = adicao_historico(msg_user, historico)
      return '', historico
    msg.submit(atualizar_historico, inputs=[msg, chatbot], outputs=[msg, chatbot], queue= False)
    limpeza.click(lambda: None, inputs=None, outputs=chatbot, queue=False)

demo.queue()
demo.launch()