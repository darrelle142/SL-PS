with open('test_math2.md', 'w', encoding='utf-8') as f:
    f.write('''---
marp: true
---

<style>
.highlight-box {
  background: #eff6ff;
  border-left: 4px solid #2563eb;
  padding: 12px;
}
</style>

# Test Slide

<div class="highlight-box">

Quá trình quyết định Markov trên đồ thị động liên kết tắc nghẽn:

$$
\\mathcal{M} = (\\mathcal{S}, \\mathcal{A}, \\mathcal{P}, \\mathcal{C}, \\gamma)
$$

</div>

Inline: $e^*$ và $f_e^*$
''')
