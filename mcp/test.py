import asyncio

from langchain_mcp_adapters.tools import load_mcp_tools
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client


async def get_mcp_tools():
    # Параметры подключения к MCP-серверу
    server_params = StdioServerParameters(
        command="python3",
        args=["math_server.py"],  # путь к вашему MCP-серверу
    )

    # Установка соединения
    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            # Инициализация соединения
            await session.initialize()

            # Загрузка инструментов
            tools = await load_mcp_tools(session)

            # Вывод списка инструментов
            for tool in tools:
                print(f"Инструмент: {tool.name}")
                print(f"Описание: {tool.description}")
                print(f"Параметры: {tool.args}\n")


# Запуск
asyncio.run(get_mcp_tools())
