import json
from datetime import datetime
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.agents import create_tool_calling_agent, AgentExecutor

# --- Bases de Datos Simuladas ---
INVENTARIO_REPUESTOS = {
    "REP-001": {"nombre": "Rodamiento de bolas SKF", "ubicacion": "Estante A1", "cantidad": 5},
    "REP-002": {"nombre": "Filtro de aceite", "ubicacion": "Estante B3", "cantidad": 0},
    "REP-003": {"nombre": "Correa de transmisión", "ubicacion": "Estante C2", "cantidad": 12},
}

HISTORIAL_FALLAS = []
ORDENES_TRABAJO = []

# --- Herramientas del Agente ---

@tool
def buscar_repuesto(codigo: str) -> str:
    """Busca información detallada de un repuesto utilizando su código (ej. REP-001)."""
    print(f"\n[Acción] Ejecutando: buscar_repuesto con código '{codigo}'...")
    if codigo in INVENTARIO_REPUESTOS:
        info = INVENTARIO_REPUESTOS[codigo]
        return f"Repuesto encontrado: {info['nombre']}. Ubicación: {info['ubicacion']}."
    return f"El repuesto con código {codigo} no fue encontrado en la base de datos."

@tool
def consultar_almacen(codigo: str) -> str:
    """Consulta la disponibilidad de un repuesto en el almacén por su código (ej. REP-001)."""
    print(f"\n[Acción] Ejecutando: consultar_almacen con código '{codigo}'...")
    if codigo in INVENTARIO_REPUESTOS:
        cantidad = INVENTARIO_REPUESTOS[codigo]['cantidad']
        estado = "Disponible" if cantidad > 0 else "Agotado"
        return f"El repuesto {codigo} está {estado}. Cantidad en almacén: {cantidad}."
    return f"No se puede consultar el almacén: código {codigo} no existe."

@tool
def generar_orden_trabajo(falla: str, equipo: str) -> str:
    """Genera una orden de trabajo indicando la falla detectada y el equipo afectado."""
    print(f"\n[Acción] Ejecutando: generar_orden_trabajo para '{equipo}' por falla '{falla}'...")
    id_orden = f"OT-{len(ORDENES_TRABAJO) + 1:03d}"
    orden = {
        "id": id_orden,
        "equipo": equipo,
        "falla": falla,
        "fecha": datetime.now().isoformat(),
        "estado": "Pendiente"
    }
    ORDENES_TRABAJO.append(orden)
    return f"Orden de trabajo generada exitosamente con ID: {id_orden}."

@tool
def actualizar_historial_fallas(falla: str, equipo: str, notas: str = "") -> str:
    """Actualiza el historial de fallas de un equipo, registrando la falla y notas adicionales si las hay."""
    print(f"\n[Acción] Ejecutando: actualizar_historial_fallas para '{equipo}'...")
    registro = {
        "equipo": equipo,
        "falla": falla,
        "notas": notas,
        "fecha": datetime.now().isoformat()
    }
    HISTORIAL_FALLAS.append(registro)
    return f"Historial actualizado exitosamente para el equipo {equipo} con la falla indicada."


def main():
    # Lista de herramientas que el agente puede usar
    tools = [buscar_repuesto, consultar_almacen, generar_orden_trabajo, actualizar_historial_fallas]

    # Inicializar el LLM
    # Nota: Usamos LM Studio (asegúrate de tener el servidor local encendido en LM Studio)
    llm = ChatOpenAI(base_url="http://localhost:1234/v1", api_key="lm-studio", temperature=0)

    # Prompt base para guiar el comportamiento del agente
    prompt = ChatPromptTemplate.from_messages([
        ("system", "Eres un asistente de mantenimiento industrial altamente capacitado. "
                   "Tu trabajo es ayudar a gestionar el mantenimiento utilizando las herramientas proporcionadas. "
                   "Si te piden buscar repuestos, consultar stock, crear órdenes o registrar fallas, usa las herramientas. "
                   "Responde de manera clara, concisa y en español."),
        ("human", "{input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad"),
    ])

    # Crear el agente con soporte para herramientas
    agent = create_tool_calling_agent(llm, tools, prompt)
    agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

    print("\n" + "="*50)
    print("🔧 Agente de Mantenimiento Iniciado")
    print("="*50)
    print("Ejemplo de uso: 'Se rompió la Bomba-Centrifuga B-101. Genera una orden de trabajo, actualiza el historial de fallas y busca si hay disponibilidad del repuesto REP-001.'")
    print("Escribe 'salir' para terminar el chat.\n")
    
    while True:
        query = input("Usuario: ")
        if query.lower() in ["salir", "exit", "quit"]:
            break
            
        try:
            # Ejecutar el agente con la petición del usuario
            response = agent_executor.invoke({"input": query})
            print(f"\nAgente: {response['output']}\n")
        except Exception as e:
            print(f"\nError al procesar la solicitud: {e}\n")

if __name__ == "__main__":
    main()
