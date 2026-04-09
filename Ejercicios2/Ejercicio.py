from langchain_ollama import ChatOllama
from langchain.agents import create_agent
from langchain.messages import SystemMessage, HumanMessage, AIMessage
from langchain.tools import tool, ToolRuntime

from dataclasses import dataclass
from typing import List

from langchain.agents.middleware import HumanInTheLoopMiddleware
from langgraph.types import Command

from langgraph.checkpoint.memory import InMemorySaver

import pandas as pd
import requests


@tool
def obtener_base_datos_recomendaciones():
    """
    Devuelve la base de datos COMPLETA de destinos turísticos en formato DataFrame.
    
    Llama a esta herramienta SIEMPRE que el usuario:
    - Pida recomendaciones de viaje (por país, tipo, presupuesto, etc.)
    - Pregunte por detalles de una ciudad (comida, seguridad, mejor época)
    - Quiera comparar destinos

    COLUMNAS DISPONIBLES (todas en inglés):
    - Destination     → nombre del lugar (ciudad, isla, parque, etc.)
    - Region          → región o comunidad autónoma
    - Country         → país
    - Category        → tipo (City, Beach, Island, Mountain Range, Coastal Town, etc.)
    - Famous Foods    → platos típicos
    - Cost of Living  → Low | Medium | Medium-high | High
    - Safety          → nivel de seguridad
    - Best Time to Visit → mejor época para visitar
    - Cultural Significance → importancia cultural
    - Description     → descripción general del destino

    ⚠️  TRADUCCIÓN OBLIGATORIA ANTES DE FILTRAR:
    Los datos están en INGLÉS. Antes de buscar, traduce los términos clave del usuario.
    
    Diccionario de países (español → inglés en el dataset):
      España       → Spain
      Francia      → France
      Italia       → Italy
      Alemania     → Germany
      Portugal     → Portugal
      Grecia       → Greece
      Reino Unido  → United Kingdom
      Holanda      → Netherlands
      Bélgica      → Belgium
      Suiza        → Switzerland
      Austria      → Austria
      Croacia      → Croatia
      República Checa → Czech Republic
      Hungría      → Hungary
      Polonia      → Poland
      Noruega      → Norway
      Suecia       → Sweden
      Dinamarca    → Denmark
      Finlandia    → Finland
      Islandia     → Iceland
      Irlanda      → Ireland
      Escocia      → Scotland (busca en Region)
      Malta        → Malta

    Diccionario de categorías (español → inglés en el dataset):
      ciudad        → City
      playa         → Beach
      isla          → Island
      montaña       → Mountain Range
      pueblo costero → Coastal Town
      parque nacional → National Park
      región        → Region
      lago          → Lake
      spa / termal  → Spa Town
      fortaleza     → Fortress

    Diccionario de presupuesto:
      barato / económico / bajo coste → Cost of Living = "Low"
      precio medio / moderado         → Cost of Living = "Medium" o "Medium-high"
      caro / lujo / alto              → Cost of Living = "High"
    """
    df = pd.read_csv("destinations.csv", encoding='latin1')
    return df


@tool
def reservarHotel(nombre: str, fecha: str):
    """Esta herramienta realiza la reserva de un hotel. Recibe el nombre del hotel y la fecha de la reserva y la escribe en un documento txt.
    args: - nombre: nombre del hotel
          - fecha: fecha de la reserva
    return: un mensaje de confirmación de la reserva
    """

    with open("reservas.txt", "a") as f:
        f.write(f"Reserva en el hotel {nombre} para la fecha {fecha}\n")

    return f"Reserva confirmada en el hotel {nombre} para la fecha {fecha}."



modelo = ChatOllama(model="qwen3:8b")
agente = create_agent(
    model=modelo,
    tools=[obtener_base_datos_recomendaciones, reservarHotel],

    checkpointer=InMemorySaver(),
    middleware=[HumanInTheLoopMiddleware(
        interrupt_on={
            "reservarHotel": True
        }
    )],

    system_prompt = """
    Eres un experto asesor de viajes conversacional. Tu objetivo es ayudar a los usuarios a planificar viajes, recomendar destinos y dar detalles precisos sobre lugares específicos.

    Tienes dos herramientas:
    - 'obtener_base_datos_recomendaciones': catálogo completo de destinos turísticos.
    - 'reservarHotel': para gestionar reservas hoteleras.

    ══════════════════════════════════════════
    PROCESO DE BÚSQUEDA OBLIGATORIO (síguelo siempre en este orden)
    ══════════════════════════════════════════

    PASO 1 — TRADUCIR antes de buscar
    Antes de filtrar el dataset, convierte mentalmente los términos del usuario al inglés
    usando el diccionario incluido en la herramienta.
    Ejemplos:
        "España"      → busca Country == "Spain"
        "Italia"      → busca Country == "Italy"
        "playa"       → busca Category == "Beach"
        "económico"   → busca Cost of Living == "Low" o "Medium"
        "Málaga"      → busca Destination == "Malaga"  (sin tilde en los datos)
        "San Sebastián" → busca Destination == "San Sebastian"

    PASO 2 — FILTRAR el DataFrame recibido
    Aplica los filtros traducidos sobre las columnas correctas.
    Si no hay coincidencia exacta, prueba búsqueda parcial (contains, ignore case).

    PASO 3 — RESPONDER en español
    Usa los datos encontrados para construir una respuesta natural, atractiva y útil.
    Nunca respondas en inglés al usuario.
    No inventes información: si un destino no está en el dataset, díselo amablemente.

    ══════════════════════════════════════════
    REGLAS ADICIONALES
    ══════════════════════════════════════════

    - CONVERSACIÓN NATURAL: No sueltes los datos crudos. Redacta respuestas persuasivas
    usando Description, Safety, Cost of Living, Famous Foods, Best Time to Visit.

    - SEGUIMIENTO DE CONTEXTO: Si el usuario pregunta "¿qué se come allí?" tras una
    recomendación previa, recuerda qué destinos recomendaste y consulta su campo
    Famous Foods sin necesidad de que el usuario repita el nombre.

    - TILDES Y CARACTERES ESPECIALES: Los datos pueden tener caracteres corruptos
    (ej. "M?laga" en lugar de "Málaga"). Usa búsqueda parcial por las primeras letras
    para encontrarlos de todas formas.

    - NO INVENTES: Si el destino no aparece en el dataset, indícalo con amabilidad
    y sugiere alternativas similares que sí estén disponibles.
    """
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
                print("\n=== INTERRUPCIÓN: ¿AUTORIZAR HERRAMIENTA? ===")
                confirmacion = input("Escribe 's' para confirmar o 'n' para cancelar: ").lower()

                if confirmacion == 's':
                    decision = "approve"
                    print("Ejecutando herramienta...")
                else:
                    decision = "reject"
                    print("Operación cancelada por el usuario.")

                # Enviamos la decisión del usuario (approve o reject)
                respuesta = agente.invoke(
                    Command(resume={"decisions": [{"type": decision}]}),
                    config={"configurable": {"thread_id": "Gabrielito"}}
                )

                # Mostramos la respuesta final tras la decisión
                for msg in respuesta["messages"]:
                    msg.pretty_print()
                
                # Salimos del stream actual para no duplicar mensajes
                break

hablarConChat(agente)
