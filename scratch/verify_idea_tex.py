# scratch/verify_idea_tex.py
import re

with open('idea.tex', 'r', encoding='utf-8') as f:
    text = f.read()

print(f"Total length: {len(text)} characters")

# 1. Check begin/end environment balance
begins = re.findall(r'\\begin\{([^}]+)\}', text)
ends = re.findall(r'\\end\{([^}]+)\}', text)

print(f"Total \\begin count: {len(begins)}")
print(f"Total \\end count: {len(ends)}")

stack = []
line_no = 1
errors = []

for line_idx, line in enumerate(text.splitlines(), 1):
    # find begins
    for match in re.finditer(r'\\begin\{([^}]+)\}', line):
        env = match.group(1)
        stack.append((env, line_idx))
    # find ends
    for match in re.finditer(r'\\end\{([^}]+)\}', line):
        env = match.group(1)
        if not stack:
            errors.append(f"Line {line_idx}: Unexpected \\end{{{env}}} with empty stack")
        else:
            last_env, last_line = stack.pop()
            if last_env != env:
                errors.append(f"Line {line_idx}: Mismatched \\end{{{env}}}, expected \\end{{{last_env}}} from line {last_line}")

if stack:
    for env, l_num in stack:
        errors.append(f"Unclosed \\begin{{{env}}} from line {l_num}")

if errors:
    print(f"Found {len(errors)} environment errors:")
    for err in errors[:10]:
        print("  ", err)
else:
    print("ALL LaTeX environments are 100% perfectly balanced!")

# 2. Check citation integrity
cites = set()
for match in re.finditer(r'\\cite\{([^}]+)\}', text):
    for k in match.group(1).split(','):
        cites.add(k.strip())

bibs = set(re.findall(r'\\bibitem\{([^}]+)\}', text))

print(f"Total cited keys: {len(cites)}")
print(f"Total bibitems: {len(bibs)}")
missing = cites - bibs
if missing:
    print("MISSING BIBITEMS:", missing)
else:
    print("ALL cited bibliography keys exist in \\begin{thebibliography}!")

# 3. Check for any remaining occurrences of problematic phrases
problematic = [
    'T_S1',
    'Radiusps',
    'RadiussL',
    'exact 0.0%',
    'purely a fortuitous topological artifact',
    'Oblivious Boundary Hugger'
]

for p in problematic:
    c = text.count(p)
    print(f"Occurrence of '{p}': {c}")
