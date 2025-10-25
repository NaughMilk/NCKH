#!/usr/bin/env python3
"""Fix indentation errors in g_training.py"""

def fix_indentation():
    with open('sections_g/g_training.py', 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    # Fix specific problematic lines
    fixes = [
        (87, '        else:', '    else:'),  # Line 88 (0-indexed)
        (88, '        opt = torch.optim.AdamW', '        opt = torch.optim.AdamW'),  # Line 89
    ]
    
    for line_idx, old_pattern, new_pattern in fixes:
        if line_idx < len(lines):
            line = lines[line_idx]
            if old_pattern in line:
                lines[line_idx] = new_pattern + '\n'
                print(f"Fixed line {line_idx + 1}")
    
    with open('sections_g/g_training.py', 'w', encoding='utf-8') as f:
        f.writelines(lines)
    
    print("Indentation fixes applied")

if __name__ == "__main__":
    fix_indentation()















