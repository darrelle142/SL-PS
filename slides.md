---
marp: true
theme: default
paginate: true
size: 16:9
math: katex
style: |
  @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&family=Fira+Code:wght@400;500&display=swap');

  :root {
    --color-background: #ffffff;
    --color-foreground: #1e293b;
    --color-heading: #0f172a;
    --color-accent: #2563eb;
    --color-accent-light: #eff6ff;
    --color-border: #e2e8f0;
    --color-danger: #dc2626;
    --color-danger-light: #fef2f2;
    --color-success: #16a34a;
    --font-default: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
  }

  section {
    background-color: var(--color-background);
    color: var(--color-foreground);
    font-family: var(--font-default);
    font-size: 21px;
    line-height: 1.6;
    padding: 44px 56px;
    border-top: 6px solid var(--color-accent);
  }

  h1, h2, h3 {
    color: var(--color-heading);
    font-weight: 700;
  }

  h1 {
    font-size: 38px;
    line-height: 1.25;
    margin-bottom: 14px;
  }

  h2 {
    font-size: 27px;
    margin-bottom: 16px;
    padding-bottom: 8px;
    border-bottom: 2px solid var(--color-accent);
  }

  h3 {
    font-size: 19px;
    color: var(--color-accent);
    margin-top: 8px;
    margin-bottom: 6px;
  }

  ul, ol {
    margin-top: 6px;
    margin-bottom: 8px;
    padding-left: 24px;
  }

  li {
    margin-bottom: 6px;
  }

  strong {
    color: var(--color-heading);
    font-weight: 600;
  }

  table {
    width: 100%;
    border-collapse: collapse;
    margin-top: 10px;
    margin-bottom: 10px;
    font-size: 17px;
  }

  th {
    background-color: #f8fafc;
    color: var(--color-heading);
    font-weight: 600;
    border-bottom: 2px solid var(--color-border);
    padding: 10px 12px;
    text-align: left;
  }

  td {
    padding: 8px 12px;
    border-bottom: 1px solid var(--color-border);
  }

  .highlight-box {
    background-color: var(--color-accent-light);
    border-left: 5px solid var(--color-accent);
    padding: 12px 18px;
    border-radius: 0 8px 8px 0;
    margin: 10px 0;
    font-size: 19px;
  }

  .danger-box {
    background-color: var(--color-danger-light);
    border-left: 5px solid var(--color-danger);
    padding: 12px 18px;
    border-radius: 0 8px 8px 0;
    margin: 10px 0;
    font-size: 19px;
  }

  .grid-2 {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 24px;
    align-items: start;
  }

  footer {
    font-size: 13px;
    color: #94a3b8;
    position: absolute;
    bottom: 16px;
    left: 56px;
    right: 56px;
  }

  section.lead {
    display: flex;
    flex-direction: column;
    justify-content: center;
    border-top: 8px solid var(--color-accent);
  }
---

<!-- _class: lead -->
# Spectral-Laplacian Prioritized Sweeping (SL-PS)
### Định tuyến Tác tử Thông minh trên Đồ thị Động Ghép nối Tắc nghẽn

**Báo cáo Ý tưởng Nghiên cứu (5 Slides)**  
Nhóm nghiên cứu | Bộ môn Khoa học Máy tính & Trí tuệ Nhân tạo

---

## 1. Hình thức hóa Bài toán: CC-DG-MDP

<div class="highlight-box">

**Congestion-Coupled Dynamic Graph MDP (CC-DG-MDP):**
Quá trình quyết định Markov trên đồ thị động liên kết tắc nghẽn:

$$\mathcal{M} = (\mathcal{S}, \mathcal{A}, \mathcal{P}, \mathcal{C}, \gamma), \quad c(s, a) = \ell_e(f_e^*(E_t)) + \delta_{\text{detour}} \cdot \mathbb{I}_{a \in \{a_{\text{up}}, a_{\text{left}}\}} + \xi_t$$

</div>

<div class="grid-2">
<div>

### Các thành phần chính
- Dòng giao thông cân bằng $f_e^*(E_t)$ giải bài toán tối ưu thế Beckmann dưới topo động $E_t = E_0 \cup \{e^*\}$:

  $$f^* = \arg\min_{f \in \mathcal{K}} \sum_{e \in E_t} \int_0^{f_e} \ell_e(x) \, dx$$

- $\delta_{\text{detour}}$: Chi phí phạt rẽ ngược hướng đích.
- $\xi_t \sim \mathcal{N}(0, \sigma^2)$: Nhiễu động quan sát ngẫu nhiên.

</div>
<div>

<div class="danger-box">

**Bản chất Thách thức:**
- Khi có cạnh mới $e^*$ xuất hiện, dòng cân bằng $f_e^*$ tự động tái phân bổ làm thay đổi độ trễ toàn mạng.
- **Nghịch lý Braess**: Lối đi tắt tưởng chừng có lợi ($c \approx 0$) lại dồn xe gây tắc nghẽn nghiêm trọng ở hạ lưu!

</div>

</div>
</div>

---

## 2. Hạn chế của RL Truyền thống & Ý tưởng Phổ Đồ thị

<div class="grid-2">
<div>

### Tại sao Prioritized Sweeping Thất bại?
- Standard-PS (Peng 1993) chỉ cập nhật số dư Bellman cục bộ từng bước ($K=15$).
- Tác tử nhìn thấy lối vào đường tắt chi phí $0$, nhưng **không kịp truyền độ trễ tắc nghẽn hạ lưu ngược về vị trí hiện tại**.
- **Hậu quả**: Tác tử lao vào bẫy nghẽn tới **50.0% số lần gặp**!

<div class="danger-box">

**Nghịch lý**: Muốn né bẫy, Standard-PS phải tăng ngân sách $K \ge 100$ (+567% compute).

</div>

</div>
<div>

### Ý tưởng Cốt lõi: Nhận thức Phổ Toàn cục
- Làm sao để tác tử "cảm nhận" được ùn tắc phía trước mà không cần duyệt cạn đồ thị?
- Sử dụng ma trận bước ngẫu nhiên $P_{\text{rw}} = D^{-1}A$ và **Toán tử Green**:

  $$\mathcal{G}_{\text{out}} = (I - P_{\text{rw}} + \eta I)^{-1}$$

- **Ý nghĩa trực quan**: Đóng vai trò như một **bộ lọc không gian (spatial filter)**: khuếch tán biến động độ trễ $\Delta \ell$ từ nút nghẽn ngược dòng về tác tử **ngay lập tức (*zero-shot*)**.

</div>
</div>

---

## 3. Giải pháp Đề xuất: Toán tử Bellman Phổ-Laplacian ($\mathcal{T}_{\text{SL}}$)

### Công thức Toán tử $\mathcal{T}_{\text{SL}}$ Tích hợp Thế hiệu Phổ

$$\mathcal{T}_{\text{SL}} V(s) = \min_{a \in \mathcal{A}(s)} \left[ c(s, a) + \gamma \sum_{s'} \mathcal{P}(s' \mid s, a) V(s') + \beta \Phi_{\text{spec}}(s') \right]$$

<div class="grid-2">
<div>

### Cơ chế Hoạt động Trực quan
- $\Phi_{\text{spec}} = \mathcal{G}_{\text{out}} |B \Delta \vec{\ell}|$ là **Thế hiệu phạt tắc nghẽn phổ**.
- Tạo ra một "hàng rào đẩy vô hình" bao quanh các khu vực sắp quá tải lưu lượng.
- Hệ số $\beta$ tự động điều chỉnh theo độ kết nối đại số của mạng lưới.

</div>
<div>

<div class="highlight-box">

**Hiệu quả Né bẫy Tức thì:**
- Khi gặp đường tắt lừa đảo, khoản phạt phổ khiến giá trị hành động vào bẫy bị đẩy lên cao:

  $$Q^*_{\text{SL}}(u, a_{\text{chord}}) > Q^*_{\text{SL}}(u, a_{\text{bypass}})$$

- Tác tử **chủ động né bẫy ngay lập tức** mà không cần tăng ngân sách quét Bellman.
- Bảo đảm toán học tính co ngặt và hội tụ ổn định.

</div>

</div>
</div>

---

## 4. Bằng chứng Thực nghiệm & Kết luận

<div class="grid-2">
<div>

### Thực nghiệm Đối sánh ($G_{3,6}$, 30 Episodes)

| Thuật toán | Ngân sách ($K$) | Độ trễ (s) | Tỷ lệ Sập bẫy |
| :--- | :---: | :---: | :---: |
| Standard-PS | 15 | $12.42 \pm 3.62$ | $50.0\%$ |
| Standard-PS (Lớn) | 100 | $11.61 \pm 2.46$ | $28.6\%$ |
| **SL-PS (Đề xuất)** | **15** | $\mathbf{11.03 \pm 0.70}$ | $\mathbf{0.0\%}$ |

<div class="highlight-box">

**0.0% Sập bẫy Tuyệt đối** &bull; **Tiết kiệm 85% Compute** so với Standard-PS ($K=100$)

</div>

</div>
<div>

### Đóng góp Chính của Đề tài
1. **Mô hình hóa CC-DG-MDP**: Mô tả chân thực tương tác giữa tác tử và dòng xe nền Wardrop.
2. **Ý tưởng Nhận thức Phổ**: Tích hợp toán tử phổ đồ thị vào hàm Bellman để mang lại tầm nhìn toàn cục.
3. **An toàn & Tiết kiệm**: Tách rời tính an toàn khỏi ngân sách tính toán ($K=15$).

<div class="highlight-box" style="text-align: center; margin-top: 14px;">

**Cảm ơn Thầy và các bạn đã lắng nghe!**

</div>

</div>
</div>
