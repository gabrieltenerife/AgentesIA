from langchain_ollama import ChatOllama
from langchain.agents import create_agent
from langchain.messages import SystemMessage, HumanMessage, AIMessage
from langchain.tools import tool, ToolRuntime

from dataclasses import dataclass
from typing import List
import sqlite3

conn = sqlite3.connect("sports_league.sqlite")
cur = conn.cursor()


@tool
def torrente_presidente_pelicula():
    """
    
    """
    datos = "cuasimodo"
    
    return datos




modelo = ChatOllama(model="qwen3:8b")
agente = create_agent(model=modelo, tools=[torrente_presidente_pelicula])


mensaje = [
    SystemMessage(content="Eres un agente que responde preguntas sobre gatos, " \
    "utilizando la herramienta que te proporciona datos sobre gatos obtenidos de la API de Meowfacts."),
    HumanMessage(content="Dame seis facts sobre gatos."),
]

resultado = agente.invoke({"messages": mensaje})

for msg in resultado["messages"]:
    msg.pretty_print()

