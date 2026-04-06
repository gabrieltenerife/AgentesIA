from langchain_ollama import ChatOllama
from langchain.agents import create_agent
from langchain.messages import SystemMessage, HumanMessage, AIMessage
from langchain.tools import tool, ToolRuntime

from dataclasses import dataclass
from typing import List

from langchain.agents.middleware import HumanInTheLoopMiddleware
from langgraph.types import Command

from langgraph.checkpoint.memory import InMemorySaver
import requests



@tool
def torrente_presidente_pelicula(n: int):
    """Esta herramienta devuelve n facts sobre gatos, obtenidos de la API de Meowfacts.
    args: - n: número de facts a obtener
    return: una lista de facts sobre gatos

    """
    response = requests.get(f"https://meowfacts.herokuapp.com/?count={n}")
    datos = response.json()
    return datos


modelo = ChatOllama(model="qwen3:8b")
agente = create_agent(
    model=modelo,
    tools=[torrente_presidente_pelicula],

    checkpointer=InMemorySaver(),
    middleware=[HumanInTheLoopMiddleware(
        interrupt_on={
            "torrente_presidente_pelicula": True
        }
    )],

    system_prompt="Eres un agente que responde preguntas sobre gatos..."
)


def hablarConChat(agente):
    while (prompt := input("> ")) != "end":
        for paso in agente.stream(
            {
                "messages": [HumanMessage(prompt)]
            },
            stream_mode="values",
            config={"configurable": {"thread_id": "Gabrielito"}}
        ):
            ultimo_mensaje = paso["messages"][-1]

            hayRazonamiento = ""
            if hasattr(ultimo_mensaje, "additional_kwargs"):
                hayRazonamiento = ultimo_mensaje.additional_kwargs.get("reasoning_content", "")

            if hayRazonamiento:
                print("\n=== PENSANDO ===")
                print(hayRazonamiento)

            print("\n=== MENSAJE ===")
            ultimo_mensaje.pretty_print()
            if "__interrupt__" in paso:
                print("\n=== INTERRUPCIÓN ===")

                resupuesta = agente.invoke(Command(
                    resume={"decisions": [{"type": "accept"}]},
                    config={"configurable": {"thread_id": "Gabrielito"}}
                )),

                for msg in resupuesta["messages"]:
                    msg.pretty_print()


hablarConChat(agente)