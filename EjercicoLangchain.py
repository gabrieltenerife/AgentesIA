from langchain_ollama import ChatOllama
from langchain.agents import create_agent
from langchain.messages import SystemMessage, HumanMessage, AIMessage
from langchain.tools import tool, ToolRuntime

from dataclasses import dataclass
from typing import List

personajes = [
        {"nombre": "Torrente", "edad": 50},
        {"nombre": "Donald Trump", "edad": 85},
        {"nombre": "Mariano Rajoy", "edad": 65},
        {"nombre": "Dandi de Barcelona", "edad": 50},
    ]



@dataclass
class context():
    lista_personajes: List[dict]

@tool
def torrente_presidente_pelicula(contexto: ToolRuntime[context]):

    """Funcion para obtener los personajes de la película 'Torrente Presidente' esta pelicula es nueva y posiblemente el modelo no tenga
    conocimiento sobre ella. Úsala siempre que te pregunten por los actores o personajes de esta película. Solo responde a la pregunta que te hagan
    sin incluir otra informacion"""

    Personajes_runtime = contexto.context.lista_personajes

    return Personajes_runtime




modelo = ChatOllama(model="qwen3:8b")
agente = create_agent(model=modelo, tools=[torrente_presidente_pelicula], context_schema=context)


mensaje = [
    SystemMessage(content="Eres un agente de inteligencia artificial que responde preguntas y también puedes ejecutar herramientas para obtener información adicional."),
    HumanMessage(content="Que edad tiene el Dandi de Barcelona en la película Torrente Presidente?"),
]

resultado = agente.invoke({"messages": mensaje}, context=context(lista_personajes=[personajes]))

print(resultado["messages"][-1].content)

