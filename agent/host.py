"""
Simple Host
"""
import json
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

from dotenv import load_dotenv
from openai import OpenAI


def _get_project_root() -> Path:
    current_file = Path(__file__).resolve()
    return current_file.parent.parent

try:
    from .client import MCPClient
except ImportError:
    import sys
    project_root = _get_project_root()
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
    from agent.client import MCPClient

# 加载 .env 文件
load_dotenv()

MODEL = "Qwen/Qwen2.5-7B-Instruct:together"

class ResearchHost:

    def __init__(
        self, 
        hf_token: Optional[str] = None, 
        model: str = MODEL,
        system_prompt: Optional[str] = None
    ) -> None:

        if hf_token is None:
            hf_token = os.environ.get("HF_TOKEN")
            if not hf_token:
                raise ValueError(
                    "HF_TOKEN not found. Please set it in environment variable "
                )
        
        self.llm_client = OpenAI(
            base_url="https://router.huggingface.co/v1",
            api_key=hf_token
        )
        self.model = model
        self.system_prompt = system_prompt
        self.mcp_clients: Dict[str, MCPClient] = {}  # {server_script: MCPClient}
        self._last_server_script: Optional[Tuple[str, ...]] = None
        self._has_printed_server_info = False
        self._has_printed_separator = False

    def _convert_mcp_tool_to_openai(self, mcp_tool: Dict) -> Dict:
        """将 MCP 工具格式转换为 OpenAI 工具格式。"""
        return {
            "type": "function",
            "function": {
                "name": mcp_tool.get("name", ""),
                "description": mcp_tool.get("description", ""),
                "parameters": mcp_tool.get("inputSchema", {})
            }
        }

    async def _ensure_clients_connected(self, server_scripts: List[str]) -> None:
        """确保指定的 server 都已连接。 """
        for script in server_scripts:
            if script not in self.mcp_clients:
                client = MCPClient(script)
                await client.connect()
                self.mcp_clients[script] = client

    async def _get_tools_from_clients(self, server_scripts: List[str]) -> List[Dict]:
        """从指定的 Client 获取工具。"""
        await self._ensure_clients_connected(server_scripts)

        all_tools = []
        for script in server_scripts:
            if script in self.mcp_clients:
                client = self.mcp_clients[script]
                tools = await client.get_tools()
                for tool in tools:
                    tool['_server'] = script
                all_tools.extend(tools) # 合并工具列表

        return all_tools

    def _get_display_server(self, mcp_tools: List[Dict]) -> str:
        """获取用于显示的服务器名称。"""
        used_servers = list(set(tool.get('_server', 'unknown') for tool in mcp_tools))
        return ', '.join(used_servers) if used_servers else 'unknown'

    def _check_and_reset_print_flags(self, server_scripts: List[str]) -> None:
        """检查 server 配置是否改变，如果改变则重置打印标志。"""
        current_server_key = tuple(sorted(server_scripts))
        if self._last_server_script != current_server_key:
            self._has_printed_server_info = False
            self._has_printed_separator = False
            self._last_server_script = current_server_key

    def _print_server_info(self, server_scripts: List[str], mcp_tools: List[Dict]) -> None:
        """打印server信息。"""
        #display_server = self._get_display_server(mcp_tools)
        available_tools = [tool.get('name', 'N/A') for tool in mcp_tools]
        #print(f"Server in USE: {display_server}")
        print(f"Tool AVAILABLE: {', '.join(available_tools)}")

    def _print_client_server_mapping(self, server_scripts: List[str]) -> None:
        """打印 Client-Server 的对应关系。
        """
        for script in server_scripts:
            if script in self.mcp_clients:
                client = self.mcp_clients[script]
                print(f"Server: {script}")
                print(f"  └─ Client Instance: {id(client)} (Memory Address)")
                print(f"  └─ Client Status: {'Connected' if client._initialized else 'Not Connected'}")
                print()

    def _extract_tool_call_info(self, tool_call) -> Tuple[str, Dict]:
        """从 tool_call 中提取工具名称和参数。
        """
        if hasattr(tool_call.function, 'name'):
            tool_name = tool_call.function.name
        else:
            tool_name = tool_call.function.get('name', '')

        if hasattr(tool_call.function, 'arguments'):
            arguments_str = tool_call.function.arguments
        else:
            arguments_str = tool_call.function.get('arguments', '{}')

        arguments = json.loads(arguments_str) if isinstance(arguments_str, str) else arguments_str
        return tool_name, arguments

    async def run_experiment(
        self,
        user_query: str,
        server_script: Optional[Union[str, List[str]]] = None
    ) -> Dict:
        """运行实验。"""

        # 如果没有提供 server_script，使用默认值
        if server_script is None:
            project_root = _get_project_root()
            server_script = str(project_root / "agent" / "server.py")
        
        # 统一 server_script 为列表格式
        if isinstance(server_script, str):
            server_scripts = [server_script]
        else:
            server_scripts = server_script

        # 检查并重置打印标志
        self._check_and_reset_print_flags(server_scripts)

        # 从指定的 Client 获取工具（通过 Client 控制）
        mcp_tools = await self._get_tools_from_clients(server_scripts)

        # 只在第一次或 server 配置改变时打印服务器信息
        if not self._has_printed_server_info:
            self._print_client_server_mapping(server_scripts)
            self._print_server_info(server_scripts, mcp_tools)
            self._has_printed_server_info = True

        # 转换为 OpenAI 工具格式
        openai_tools = [self._convert_mcp_tool_to_openai(tool) for tool in mcp_tools]

        # 构建 messages 列表
        messages = []
        # 如果设置了 system_prompt，添加到 messages 开头
        if self.system_prompt:
            messages.append({"role": "system", "content": self.system_prompt})
        # 添加 user message
        messages.append({"role": "user", "content": user_query})

        # 调用 LLM API
        completion = self.llm_client.chat.completions.create(
            model=self.model,
            messages=messages,
            tools=openai_tools,
            tool_choice="auto"
        )

        # 分析结果
        message = completion.choices[0].message
        display_server = self._get_display_server(mcp_tools)

        result = {
            "user_query": user_query,
            "available_tools": [tool.get('name', 'N/A') for tool in mcp_tools],
            "selected_server": display_server,
            "selected_tool": None,
            "tool_arguments": None,
            "llm_response": message.content,
            "tool_calls": []
        }

        # 处理工具调用（只提取选择结果，不执行工具）
        # hasattr: 检查对象是否有指定属性或方法
        if hasattr(message, 'tool_calls') and message.tool_calls:
            for tool_call in message.tool_calls:
                tool_name, arguments = self._extract_tool_call_info(tool_call)

                # 找到工具对应的 server
                tool_server = None
                for tool in mcp_tools:
                    if tool.get('name') == tool_name:
                        tool_server = tool.get('_server')
                        break

                # 记录选择结果（研究重点：schema 对 tool selection 的影响）
                result["tool_calls"].append({
                    "tool_name": tool_name,
                    "arguments": arguments,
                    "server": tool_server
                })
                result["selected_tool"] = tool_name
                result["tool_arguments"] = arguments

                # 工具执行已移除，只研究 tool selection。如果需要执行工具，可以取消下面的注释：
                # if tool_server and tool_server in self.mcp_clients:
                #     client = self.mcp_clients[tool_server]
                #     tool_result = await client.call_tool(tool_name, arguments)

        return result

    def print_result(self, result: Dict) -> None:
        """打印实验结果。
        """
        # 只在第一次Result打印分隔符
        if not self._has_printed_separator:
            print("=" * 60)
            print("Results")
            print("=" * 60)
            self._has_printed_separator = True

        print(f"User Query: {result['user_query']}")

        if result['selected_tool']:
            print(f"✅ LLM Choose: {result['selected_tool']}")
            #print(f"Parameters Schema: {json.dumps(result['tool_arguments'], indent=2, ensure_ascii=False)}")

            # 显示工具调用信息（架构完整性验证）
            if result.get('tool_calls'):
                for tool_call_info in result['tool_calls']:
                    if tool_call_info.get('tool_name') == result['selected_tool']:
                        server = tool_call_info.get('server', 'unknown')
                        print(f"Tool from Server: {server}")
                        break
        else:
            print("❌ LLM didn't choose any tool")

        if result['llm_response']:
            print(f"LLM Response: {result['llm_response']}")
        print("=" * 60)

    async def cleanup(self) -> None:
        """清理所有客户端连接。"""
        for script, client in list(self.mcp_clients.items()):
            try:
                await client.disconnect()
            except Exception:
                pass
        self.mcp_clients.clear()

