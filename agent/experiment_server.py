"""
Experiment Server
"""
from fastmcp import FastMCP


mcp = FastMCP("Experiment Server")

# 所有工具使用相同的描述
COMMON_DESCRIPTION = "This is a general-purpose data processing tool that can handle various types of input data"


# 工具1: 1个参数
@mcp.tool(description=COMMON_DESCRIPTION)
def tool_xy87(data_input: str) -> str:
    return f"处理数据: {data_input}"


# 工具2: 2个参数
@mcp.tool(description=COMMON_DESCRIPTION)
def tool_ty32(content: str, mode: str) -> str:
    return f"处理内容: {content}, 模式: {mode}"


# 工具3: 3个参数
@mcp.tool(description=COMMON_DESCRIPTION)
def tool_nt68(text: str, count: int, format_type: str) -> str:
    return f"执行文本: {text}, 数量: {count}, 格式: {format_type}"


# 工具4: 4个参数
@mcp.tool(description=COMMON_DESCRIPTION)
def tool_lk72(value: str, threshold: int, category: str, enable_cache: bool) -> str:
    return f"操作值: {value}, 阈值: {threshold}, 类别: {category}, 缓存: {enable_cache}"

if __name__ == "__main__":
    mcp.run()

