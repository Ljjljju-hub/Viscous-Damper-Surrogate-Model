input_file = "Case_0050.vtu"
output_file = "output_v2.vtu"

with open(input_file, 'r', encoding='utf-8') as f:
    lines = f.readlines()

out_lines = []
for line in lines:
    if line.lstrip().startswith('<'):
        out_lines.append(line)   # 保留标签行
    # 其他行（数值行）直接丢弃

with open(output_file, 'w', encoding='utf-8') as f:
    f.writelines(out_lines)

print("方案二完成，只保留标签，所有数值行被删除")