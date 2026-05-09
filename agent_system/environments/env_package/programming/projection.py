import re
from typing import List


def programming_projection(actions: List[str]):
    parsed = []
    valids = []

    for a in actions:
        text = a.strip()
        code = ""

        # 1. 优先解析 <code>...</code>
        start = text.find("<code>")
        end = text.find("</code>")
        if start != -1 and end != -1 and end > start:
            code = text[start + len("<code>"):end].strip()

        # 2. 兼容模型不听格式，输出 ```python ... ```
        if not code:
            m = re.search(r"```(?:python)?\s*(.*?)```", text, flags=re.DOTALL)
            if m:
                code = m.group(1).strip()

        # 3. 去掉残留 markdown fence
        if code.startswith("```python"):
            code = code[len("```python"):].strip()
        elif code.startswith("```"):
            code = code[len("```"):].strip()

        if code.endswith("```"):
            code = code[:-3].strip()

        # 4. 过滤明显无效输出
        if code and ("def " in code or "import " in code or "class " in code):
            parsed.append(code)
            valids.append(1)
        else:
            parsed.append("")
            valids.append(0)

    return parsed, valids