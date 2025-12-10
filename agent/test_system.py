"""
系统测试脚本 - 测试 MCP Client 和 Host 的功能
"""
import asyncio
from pathlib import Path

try:
    from .client import MCPClient
    from .host import ResearchHost
except ImportError:
    # 如果相对导入失败（直接运行时），使用绝对导入
    import sys
    current_file = Path(__file__).resolve()
    project_root = current_file.parent.parent
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
    from agent.client import MCPClient
    from agent.host import ResearchHost


def _get_project_root() -> Path:
    current_file = Path(__file__).resolve()
    return current_file.parent.parent


async def test_client() -> None:
    """测试函数：连接所有服务器并列出其工具。"""
    project_root = _get_project_root()
    
    # 定义所有服务器（相对于项目根目录的路径）
    server_relative_paths = ["agent/server.py", "agent/experiment_server.py"]
    servers = [str(project_root / path) for path in server_relative_paths]

    for server_script in servers:
        print(f"\nServer: {server_script}")
        print("-" * 60)

        client = MCPClient(server_script)
        try:
            await client.connect()
            tools = await client.get_tools()
            print(f"Find {len(tools)} Tools:") # Find x tools
            for tool in tools:
                print(f"  - {tool.get('name', 'N/A')} : {tool.get('description', 'N/A')}")
                print(f"  {tool.get('inputSchema', 'N/A')}")
            print("=" * 60)
            await client.disconnect()
        except Exception as e:
            print(f"  ❌ 连接失败: {e}")
            try:
                await client.disconnect()
            except Exception:
                pass


async def test_host() -> None:
    host = None
    try:
        project_root = _get_project_root()
        host = ResearchHost(system_prompt="You are a professional tool selection assistant")
    except ValueError as e:
        print(f"Error: {e}")
        return

    try:
        test_queries = [
            "Find information about USDT",
            "Search for the latest financial news, limiting the results to 5 items, sorted from newest to oldest",
            "Get system status"
        ]

        # default server
        print("\n=== Example1: default server ===")
        default_server = str(project_root / "agent" / "server.py")
        for query in test_queries:
            result = await host.run_experiment(query, server_script=default_server)
            host.print_result(result)

        # selected server
        print("\n=== Example2: selected server ===")
        experiment_server = str(project_root / "agent" / "experiment_server.py")
        result = await host.run_experiment(
            test_queries[0],
            server_script=experiment_server
        )
        host.print_result(result)

        # multiple server
        print("\n=== Example3: multiple server ===")
        server1 = str(project_root / "agent" / "server.py")
        server2 = str(project_root / "agent" / "experiment_server.py")
        result = await host.run_experiment(
            test_queries[0],
            server_script=[server1, server2]
        )
        host.print_result(result)
    finally:
        # 确保清理所有客户端连接
        if host:
            await host.cleanup()


async def main() -> None:
    print("=" * 60)
    print("MCP System Tests")
    print("=" * 60)
    
    print("\n[1/2] Testing MCP Client...")
    await test_client()
    
    print("\n[2/2] Testing Research Host...")
    await test_host()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nTest interrupted by user")
    except Exception as e:
        print(f"\nTest failed with error: {e}")
        import traceback
        traceback.print_exc()

