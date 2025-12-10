"""
MCP Client
"""
import asyncio
import sys
from typing import Dict, List, Optional

from mcp.client.session import ClientSession
from mcp.client.stdio import stdio_client, StdioServerParameters


class MCPClient:

    def __init__(self, server_script: str) -> None:
        """初始化 MCP 客户端。"""
        self.server_script = server_script
        self._session: Optional[ClientSession] = None
        self._stdio_context = None
        self._session_context = None
        self._tools_cache: Optional[List[Dict]] = None
        self._initialized = False

    async def connect(self) -> None:
        """连接到 MCP server并初始化。"""
        if self._session is not None:
            return  # 已经连接

        try:
            # 创建 stdio 传输参数
            server_params = StdioServerParameters(
                command=sys.executable,
                args=[self.server_script]
            )

            # Start the server subprocess
            self._stdio_context = stdio_client(server_params)
            read, write = await self._stdio_context.__aenter__()

            # 创建client session
            self._session_context = ClientSession(read, write)
            self._session = await self._session_context.__aenter__()

            # 初始化session
            await self._session.initialize()
            self._initialized = True

        except Exception as e:
            # 清理资源
            await self._cleanup()
            raise Exception(f"Failed to connect to MCP server: {e}")

    async def _cleanup(self) -> None:
        """清理资源。"""
        # 先清理 session，再清理 stdio
        # 顺序很重要：先关闭内层上下文，再关闭外层上下文
        if self._session_context is not None:
            session_context = self._session_context
            self._session_context = None
            self._session = None
            try:
                await session_context.__aexit__(None, None, None)
            except (asyncio.CancelledError, GeneratorExit, RuntimeError, Exception):
                # 忽略清理时的所有异常，这些通常是由于任务取消或上下文管理器生命周期问题导致的
                # 特别是在程序退出时，asyncio 会取消所有任务，导致这些异常
                pass

        if self._stdio_context is not None:
            stdio_context = self._stdio_context
            self._stdio_context = None
            try:
                await stdio_context.__aexit__(None, None, None)
            except (asyncio.CancelledError, GeneratorExit, RuntimeError, Exception):
                # 忽略清理时的所有异常，这些通常是由于任务取消或上下文管理器生命周期问题导致的
                # 特别是在程序退出时，asyncio 会取消所有任务，导致这些异常
                pass

        self._initialized = False
        self._tools_cache = None

    async def get_tools(self) -> List[Dict]:
        """获取工具列表。"""
        if not self._initialized or self._session is None:
            raise Exception("Client is not connected. Call connect() first.")

        if self._tools_cache is None:
            try:
                # 调用官方 SDK 的 list_tools 方法
                result = await self._session.list_tools()

                # 将 SDK 返回的工具对象转换为字典格式
                self._tools_cache = []
                if result.tools:
                    for tool in result.tools:
                        # inputSchema 可能是 Pydantic (官方 SDK) or dict
                        if tool.inputSchema:
                            if hasattr(tool.inputSchema, 'model_dump'):
                                input_schema = tool.inputSchema.model_dump()
                            elif isinstance(tool.inputSchema, dict):
                                input_schema = tool.inputSchema
                            else:
                                input_schema = {}
                        else:
                            input_schema = {}
                        
                        tool_dict = {
                            "name": tool.name,
                            "description": tool.description or "",
                            "inputSchema": input_schema
                        }
                        self._tools_cache.append(tool_dict)

            except Exception as e:
                raise Exception(f"Failed to get tools: {e}")

        return self._tools_cache

    async def call_tool(self, tool_name: str, arguments: Dict) -> Dict:
        """调用工具并返回执行结果。"""
        if not self._initialized or self._session is None:
            raise Exception("Client is not connected. Call connect() first.")

        try:
            # 调用官方 SDK 的 call_tool 方法
            result = await self._session.call_tool(tool_name, arguments or {})

            # 将 SDK 返回的结果转换为字典格式
            # CallToolResult 包含 content 字段，需要转换为原来的格式
            result_dict = {
                "content": []
            }

            if result.content:
                for item in result.content:
                    if hasattr(item, 'text'):
                        result_dict["content"].append({
                            "type": "text",
                            "text": item.text
                        })
                    elif hasattr(item, 'image'):
                        result_dict["content"].append({
                            "type": "image",
                            "data": item.image.data,
                            "mimeType": item.image.mimeType
                        })

            # 如果有 isError，也包含进去
            if hasattr(result, 'isError') and result.isError:
                result_dict["isError"] = True

            return result_dict

        except Exception as e:
            raise Exception(f"Failed to call tool '{tool_name}': {e}")

    async def disconnect(self) -> None:
        """关闭连接并清理资源。"""
        await self._cleanup()
