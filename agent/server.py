"""
MCP Server
"""
from typing import Dict, List

from fastmcp import FastMCP


mcp = FastMCP("Test Server")


# 工具1: 简单的 inputSchema - 只有一个字符串参数
@mcp.tool(description="Simple query tool that only requires a query string")
def simple_query(query: str) -> str:
    return f"处理查询: {query}"


# 工具2: 中等复杂度的 inputSchema - 多个基本类型参数
@mcp.tool(description="Search tool with filter parameters including query, limit, and sort options")
def search_with_filters(
    query: str,
    limit: int = 10,
    sort_by: str = "relevance"
) -> str:
    return f"搜索 '{query}'，限制 {limit} 条，排序方式: {sort_by}"


# 工具3: 复杂的 inputSchema - 包含嵌套对象和数组
@mcp.tool(description="Advanced search tool with complex parameter structure including nested objects and arrays")
def advanced_search(
    query: str,
    filters: Dict,
    options: List[str],
    max_results: int = 20
) -> str:
    return f"高级搜索 '{query}'，过滤器: {filters}，选项: {options}，最大结果: {max_results}"


# 工具4: 非常简单的 inputSchema - 无参数
@mcp.tool(description="Get system status, requires no parameters")
def get_status() -> str:
    return "系统状态正常"

if __name__ == "__main__":
    mcp.run()

