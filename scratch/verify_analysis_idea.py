# scratch/verify_analysis_idea.py
import re

with open('analysis_idea.tex', 'r', encoding='utf-8') as f:
    text = f.read()

print(f"Total length: {len(text)} characters")

# 1. Environment balance
begins = re.findall(r'\\begin\{([^}]+)\}', text)
ends = re.findall(r'\\end\{([^}]+)\}', text)

print(f"Total \\begin count: {len(begins)}")
print(f"Total \\end count: {len(ends)}")

stack = []
errors = []

for line_idx, line in enumerate(text.splitlines(), 1):
    for match in re.finditer(r'\\begin\{([^}]+)\}', line):
        env = match.group(1)
        stack.append((env, line_idx))
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

# 2. Citations check
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

# 3. Check Section headers
sections = re.findall(r'\\section\*?\{([^}]+)\}', text)
print("\nSections in analysis_idea.tex:")
for idx, s in enumerate(sections, 1):
    print(f"  {idx}. {s}")

# 4. Check for empirical artifacts
empirical_keywords = [
    '0/30',
    '0/180',
    'Table~\\ref{tab:results}',
    'Figure~\\ref{fig:benchmark',
    'Figure~\\ref{fig:ablation',
    'Figure~\\ref{fig:stochastic',
    'Table~\\ref{tab:hyperparams}',
    'Wilson score interval of [0.0%',
    '12.42s',
    '11.17s',
    'Oblivious Boundary Hugger',
    'Greedy-Dijkstra'
]

print("\nEmpirical keyword search in analysis_idea.tex:")
for kw in empirical_keywords:
    c = text.count(kw)
    print(f"  '{kw}': {c}")
