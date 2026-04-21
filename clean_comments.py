import ast
import sys

def remove_docstrings_and_comments(source_code):
    tree = ast.parse(source_code)
    
    lines = source_code.split('\n')
    result_lines = []
    in_docstring = False
    docstring_char = None
    
    i = 0
    while i < len(lines):
        line = lines[i]
        stripped = line.lstrip()
        
        if not in_docstring:
            if '"""' in line or "'''" in line:
                if '"""' in line:
                    count = line.count('"""')
                    if count == 2:
                        comment_idx = line.find('#')
                        if comment_idx == -1:
                            i += 1
                            continue
                        else:
                            result_lines.append(line[comment_idx:])
                            i += 1
                            continue
                    elif count == 1:
                        in_docstring = True
                        docstring_char = '"""'
                        i += 1
                        continue
                elif "'''" in line:
                    count = line.count("'''")
                    if count == 2:
                        comment_idx = line.find('#')
                        if comment_idx == -1:
                            i += 1
                            continue
                        else:
                            result_lines.append(line[comment_idx:])
                            i += 1
                            continue
                    elif count == 1:
                        in_docstring = True
                        docstring_char = "'''"
                        i += 1
                        continue
            
            if stripped.startswith('#'):
                i += 1
                continue
            
            comment_idx = line.find('#')
            if comment_idx != -1:
                in_string = False
                quote_char = None
                j = 0
                while j < comment_idx:
                    if not in_string:
                        if line[j] in ('"', "'"):
                            in_string = True
                            quote_char = line[j]
                    else:
                        if line[j] == quote_char and (j == 0 or line[j-1] != '\\'):
                            in_string = False
                    j += 1
                
                if not in_string:
                    line = line[:comment_idx].rstrip()
            
            if line.strip():
                result_lines.append(line)
            elif not result_lines or result_lines[-1].strip():
                pass
        else:
            if docstring_char in line:
                in_docstring = False
                docstring_char = None
        
        i += 1
    
    while result_lines and not result_lines[-1].strip():
        result_lines.pop()
    
    return '\n'.join(result_lines) + '\n'

if __name__ == '__main__':
    if len(sys.argv) != 3:
        print("Usage: python clean_comments.py <input_file> <output_file>")
        sys.exit(1)
    
    input_file = sys.argv[1]
    output_file = sys.argv[2]
    
    with open(input_file, 'r') as f:
        source = f.read()
    
    cleaned = remove_docstrings_and_comments(source)
    
    with open(output_file, 'w') as f:
        f.write(cleaned)
    
    print(f"Cleaned {input_file} -> {output_file}")
