import json

path = "/Users/joaoquintanilha/.claude/projects/-Users-joaoquintanilha-Downloads-data-297-final-paper-full-project/6a251319-faef-4972-b392-b9255f8964d1.jsonl"
out = "/Users/joaoquintanilha/Downloads/data_297_final_paper/full_project/claude_session.md"

lines = []
with open(path) as f:
    for line in f:
        try:
            lines.append(json.loads(line.strip()))
        except Exception:
            pass

with open(out, "w") as f:
    f.write("# Claude Code Session — Beyond WhAM\n\n")
    for entry in lines:
        msg = entry.get("message", {})
        role = msg.get("role", "")
        content = msg.get("content", "")
        if not role or not content:
            continue
        if isinstance(content, list):
            parts = []
            for block in content:
                if not isinstance(block, dict):
                    continue
                btype = block.get("type", "")
                if btype == "text":
                    parts.append(block.get("text", ""))
                elif btype == "tool_use":
                    name = block.get("name", "")
                    inp = json.dumps(block.get("input", {}), indent=2)
                    parts.append(f"**[tool: {name}]**\n```json\n{inp}\n```")
                elif btype == "tool_result":
                    result = block.get("content", "")
                    if isinstance(result, list):
                        result = "\n".join(r.get("text", "") for r in result if isinstance(r, dict))
                    parts.append(f"**[tool result]**\n```\n{result}\n```")
            content = "\n\n".join(p for p in parts if p.strip())
        if not content.strip():
            continue
        label = "## Human" if role == "user" else "## Assistant"
        f.write(f"{label}\n\n{content}\n\n---\n\n")

print(f"Done — {len(lines)} entries written to {out}")
