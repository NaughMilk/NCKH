#!/usr/bin/env python3
"""Fix all indentation errors in g_training.py"""

def fix_all_indentation():
    with open('sections_g/g_training.py', 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    # Fix specific problematic lines
    fixes = [
        (222, '        if val_loss < best_val:', '            if val_loss < best_val:'),  # Line 223 (0-indexed)
    ]
    
    # Also fix any lines that have 8 spaces at the beginning when they should have 12
    for i, line in enumerate(lines):
        if line.startswith('        ') and not line.startswith('            '):
            # Check if this line should be indented with 12 spaces instead of 8
            if any(keyword in line for keyword in ['if val_loss < best_val:', 'if val_loss < best_val:', 'if val_loss < best_val:']):
                lines[i] = '            ' + line[8:]  # Replace 8 spaces with 12
                print(f"Fixed line {i + 1}: {line.strip()}")
    
    with open('sections_g/g_training.py', 'w', encoding='utf-8') as f:
        f.writelines(lines)
    
    print("All indentation fixes applied")

if __name__ == "__main__":
    fix_all_indentation()















