#!/usr/bin/env python3
"""Fix all indentation errors in g_training.py"""

def fix_all_indentation():
    with open('sections_g/g_training.py', 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    # Fix specific problematic lines
    fixes = [
        (87, '        else:', '    else:'),  # Line 88 (0-indexed)
        (88, '        opt = torch.optim.AdamW', '        opt = torch.optim.AdamW'),  # Line 89
    ]
    
    # Also fix any lines that have wrong indentation
    for i, line in enumerate(lines):
        # Fix lines that should have 4 spaces but have 8
        if line.startswith('        ') and not line.startswith('            '):
            if any(keyword in line for keyword in ['else:', 'elif', 'except', 'finally', 'return', '_log_']):
                lines[i] = '    ' + line[8:]  # Replace 8 spaces with 4
                print(f"Fixed line {i + 1}: {line.strip()}")
        
        # Fix lines that should have 12 spaces but have 8
        elif line.startswith('        ') and not line.startswith('            '):
            if any(keyword in line for keyword in ['model.eval()', 'all_preds =', 'all_targets =', 'with torch.no_grad():']):
                lines[i] = '            ' + line[8:]  # Replace 8 spaces with 12
                print(f"Fixed line {i + 1}: {line.strip()}")
    
    with open('sections_g/g_training.py', 'w', encoding='utf-8') as f:
        f.writelines(lines)
    
    print("All indentation fixes applied")

if __name__ == "__main__":
    fix_all_indentation()















