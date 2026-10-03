Nếu nhìn **AI chain theo dòng tiền từ hạ tầng vật lý → compute → cloud → model → ứng dụng**, thì có thể chia thành khoảng **9 tầng chính**. Với góc nhìn đầu tư, mình sẽ ưu tiên các công ty niêm yết và ghi ticker để bạn dễ theo dõi.

### 1. Thiết kế chip & AI accelerator
Đây là tầng tạo ra GPU/ASIC/CPU dùng để train và inference AI.

| Công ty | Ticker | Vai trò |
|---|---|---|
| **NVIDIA** | NVDA | GPU AI, NVLink, networking, AI systems |
| **AMD** | AMD | Instinct GPU, EPYC CPU, Helios rack |
| **Broadcom** | AVGO | Custom AI ASIC + networking |
| **Marvell** | MRVL | Custom accelerator, optical/network chips |
| Qualcomm | QCOM | Edge AI / AI PC / mobile |
| Intel | INTC | CPU, accelerator, foundry |
| Arm | ARM | CPU architecture cho server/edge AI |

NVIDIA hiện không chỉ bán GPU mà còn cung cấp networking như NVLink, InfiniBand, Spectrum-X và BlueField, tức đang mở rộng từ **chip → toàn bộ AI factory**. AMD cũng đang đi theo hướng tương tự với GPU Instinct + EPYC + Pensando networking + Helios rack-scale systems. :chatgpt-content-reference{index="1"}

---

### 2. EDA — phần mềm thiết kế chip

Trước khi NVIDIA hay AMD có chip để bán, chip phải được thiết kế và kiểm chứng.

| Công ty | Ticker | Vai trò |
|---|---|---|
| **Synopsys** | SNPS | EDA, chip design, verification, IP |
| **Cadence** | CDNS | EDA, 3D-IC, chip/package/system design |
| Siemens EDA | SIEGY | EDA / verification |

Đây là một tầng thường bị bỏ qua nhưng rất quan trọng. Synopsys và Cadence cung cấp công cụ mà các hãng semiconductor dùng để thiết kế chip ngày càng phức tạp. :chatgpt-content-reference{index="2"}

---

### 3. Semiconductor equipment — máy để sản xuất chip

```text
AI demand
   ↓
More advanced GPUs / HBM
   ↓
More advanced semiconductor fabs
   ↓
More equipment demand
```

| Công ty | Ticker | Vai trò |
|---|---|---|
| **ASML** | ASML | EUV/DUV lithography |
| **Applied Materials** | AMAT | Deposition, materials engineering |
| **Lam Research** | LRCX | Etch/deposition |
| **KLA** | KLAC | Inspection/metrology |
| **Tokyo Electron** | 8035.T | Semiconductor equipment |

ASML cho biết tăng trưởng leading-edge logic và HBM đang được thúc đẩy đáng kể bởi AI; Applied Materials cũng đang tung thiết bị mới phục vụ DRAM/HBM và advanced packaging cho AI chip. :chatgpt-content-reference{index="3"}

Đây là nhóm mình gọi là **“picks and shovels của picks and shovels”**.

---

### 4. Foundry & advanced packaging

Các công ty thiết kế chip thường không tự sản xuất.

| Công ty | Ticker | Vai trò |
|---|---|---|
| **TSMC** | TSM | Foundry quan trọng nhất cho advanced AI chips |
| Samsung Electronics | 005930.KS | Foundry + memory |
| Intel | INTC | Intel Foundry |
| ASE Technology | ASX | Packaging/testing |
| Amkor | AMKR | Advanced packaging/testing |

Đặc biệt với AI, **advanced packaging trở thành bottleneck quan trọng**.

TSMC CoWoS tích hợp compute die với HBM và được TSMC mô tả là nền tảng quan trọng cho HPC và AI products. :chatgpt-content-reference{index="4"}

Một dependency rất đáng nhớ:

**NVDA/AMD → TSMC → CoWoS → HBM**

---

### 5. Memory — đặc biệt HBM

GPU nhanh mà không được cấp dữ liệu đủ nhanh thì cũng bị nghẽn.

| Công ty | Ticker | Vai trò |
|---|---|---|
| **SK Hynix** | 000660.KS | HBM |
| **Micron** | MU | HBM, DRAM |
| **Samsung** | 005930.KS | HBM, DRAM, NAND |

HBM hiện gần như là một trong những thành phần cốt lõi của AI accelerator. Ví dụ Micron HBM4 đạt băng thông trên 2.8 TB/s mỗi stack và được thiết kế trực tiếp cho thế hệ AI workload mới. :chatgpt-content-reference{index="5"}

---

### 6. Networking & optical

Khi một AI cluster có hàng chục nghìn GPU, GPU phải giao tiếp cực nhanh.

| Công ty | Ticker | Vai trò |
|---|---|---|
| **Broadcom** | AVGO | Ethernet switching/custom silicon |
| **Arista Networks** | ANET | AI Ethernet networking |
| **NVIDIA** | NVDA | InfiniBand, Spectrum-X, NVLink |
| Marvell | MRVL | Networking / optical |
| Coherent | COHR | Optical components |
| Lumentum | LITE | Optical connectivity |
| Fabrinet | FN | Optical manufacturing |

Arista đang phát triển 1.6T networking phục vụ rack-scale AI infrastructure; Broadcom cũng tập trung mạnh vào scale-up/scale-out networks cho AI clusters. :chatgpt-content-reference{index="6"}

Đây là lý do **ANET / AVGO / LITE / COHR** thường được thị trường xếp vào AI infrastructure chain dù họ không sản xuất GPU.

---

### 7. Servers, racks & data-center systems

Chip phải được ráp thành server/rack hoàn chỉnh.

| Công ty | Ticker | Vai trò |
|---|---|---|
| **Dell** | DELL | AI servers / Dell AI Factory |
| **Super Micro Computer** | SMCI | GPU servers/racks |
| HPE | HPE | Enterprise AI infrastructure |
| Lenovo | 0992.HK | AI servers |
| Wiwynn | 6669.TW | Hyperscale server |
| Quanta | 2382.TW | Servers |
| Foxconn | 2317.TW | AI server manufacturing |

Dell chẳng hạn đang xây cả **Dell AI Factory with NVIDIA**, tức không còn chỉ bán server mà cung cấp integrated AI infrastructure stack. :chatgpt-content-reference{index="7"}

---

### 8. Power + cooling + electrical infrastructure

Đây là nhánh rất quan trọng và có thể trở thành bottleneck lớn nhất.

```text
More GPUs
    ↓
More racks
    ↓
More MW / GW electricity
    ↓
More transformers
switchgear
UPS
cooling
generators
grid equipment
```

Các công ty đáng chú ý:

| Công ty | Ticker | Vai trò |
|---|---|---|
| **Vertiv** | VRT | Cooling + power management |
| **Eaton** | ETN | Power distribution |
| Schneider Electric | SU.PA | Electrical + cooling |
| **GE Vernova** | GEV | Power generation/grid |
| Siemens Energy | ENR.DE | Grid/power |
| Caterpillar | CAT | Backup generators |
| Cummins | CMI | Generators |
| Trane Technologies | TT | Cooling/HVAC |
| Johnson Controls | JCI | Cooling/building systems |

Đây là nhánh mà AI boom chuyển từ câu chuyện semiconductor sang:

> **“Làm sao cấp đủ điện và giải nhiệt cho tất cả GPU đó?”**

---

### 9. Data-center owners / operators

| Công ty | Ticker | Vai trò |
|---|---|---|
| Equinix | EQIX | Data centers |
| Digital Realty | DLR | Data-center REIT |
| CoreWeave | CRWV | AI GPU cloud |
| Oracle | ORCL | Cloud + AI infrastructure |

Ngoài ra Meta, Microsoft, Amazon và Google đang tự xây rất nhiều infrastructure riêng.

---

## 10. Hyperscalers / Cloud

Đây là những người **mua lượng AI infrastructure cực lớn**.

| Công ty | Ticker | AI stack |
|---|---|---|
| **Microsoft** | MSFT | Azure + OpenAI ecosystem + own silicon |
| **Amazon** | AMZN | AWS + Trainium + Inferentia |
| **Alphabet** | GOOGL | Google Cloud + TPU + Gemini |
| **Meta** | META | Huge AI infrastructure + own accelerators |
| Oracle | ORCL | OCI AI cloud |

AWS có Trainium riêng; Google đã đưa TPU thế hệ mới vào Cloud; Microsoft vừa dùng NVIDIA vừa AMD và silicon riêng. :chatgpt-content-reference{index="8"}

Đây là một điểm quan trọng:

**Hyperscaler vừa là khách hàng của NVDA, vừa có khả năng trở thành đối thủ của NVDA.**

Ví dụ:

```text
Amazon → Trainium
Google → TPU
Microsoft → Maia
Meta → MTIA
```

---

## 11. Frontier AI models

Đây là tầng sử dụng compute để tạo foundation models.

| Công ty | Public? | Sản phẩm |
|---|---|---|
| OpenAI | Private | GPT |
| Anthropic | Private | Claude |
| Google | GOOGL | Gemini |
| Meta | META | Llama |
| xAI | Private | Grok |
| Mistral | Private | Mistral |
| DeepSeek | Private | DeepSeek |

---

## 12. AI software / data infrastructure

Các công ty nằm giữa model và enterprise applications:

| Công ty | Ticker | Vai trò |
|---|---|---|
| **Palantir** | PLTR | Enterprise AI / data |
| **Snowflake** | SNOW | Data cloud / AI |
| **Datadog** | DDOG | Observability |
| **MongoDB** | MDB | Database / vector workloads |
| **Cloudflare** | NET | Edge AI infrastructure |
| ServiceNow | NOW | Enterprise agents |
| Salesforce | CRM | Agentic AI / CRM |
| Adobe | ADBE | Generative AI |

---

# Nếu gom lại thành một AI chain đơn giản

```text
EDA
SNPS / CDNS
     ↓
Semiconductor Equipment
ASML / AMAT / LRCX / KLAC
     ↓
Foundry
TSM / Samsung
     ↓
Memory + Packaging
SK Hynix / MU / Samsung / ASE
     ↓
AI Compute
NVDA / AMD / AVGO
     ↓
Networking
AVGO / ANET / NVDA / MRVL / LITE / COHR
     ↓
AI Servers
DELL / SMCI / HPE
     ↓
Power + Cooling
VRT / ETN / GEV / Schneider / CAT
     ↓
Data Centers
EQIX / DLR / CRWV
     ↓
Cloud
MSFT / AMZN / GOOGL / ORCL
     ↓
Foundation Models
OpenAI / Anthropic / Google / Meta
     ↓
Enterprise AI
PLTR / NOW / CRM / ADBE / SNOW
     ↓
End-user applications
```

### Với góc nhìn **cổ phiếu AI chain**, mình đặc biệt chú ý khoảng 20 mã này:

**Semiconductor core:**  
`NVDA • AMD • AVGO • TSM • ASML • MU`

**Equipment:**  
`AMAT • LRCX • KLAC`

**Networking / optical:**  
`ANET • MRVL • LITE • COHR`

**Server / infrastructure:**  
`DELL • SMCI`

**Power / cooling:**  
`VRT • ETN • GEV`

**Cloud:**  
`MSFT • AMZN • GOOGL`

Điểm rất đáng chú ý là **AI chain không phải một chuỗi tuyến tính**. Ví dụ NVIDIA phụ thuộc TSMC + HBM + equipment, nhưng đồng thời cạnh tranh với Broadcom/AMD và cả chip tự thiết kế của Google/Amazon/Microsoft. Chính những **dependency + bottleneck** này thường tạo ra tín hiệu sớm rất tốt để dự đoán cổ phiếu ở tầng kế tiếp. :chatgpt-content-reference{index="9"}

Nếu áp dụng đúng cách phân tích mà chúng ta đang dùng cho **TSMC / NVIDIA / ASML / VRT**, thì bước tiếp theo rất hữu ích là mình có thể vẽ cho bạn **toàn bộ AI chain thành một graph khoảng 40–50 công ty, với mũi tên “ai bán cho ai / ai là customer-supplier của ai”, và đánh dấu những công ty nào là bottleneck**. Cái đó sẽ hữu ích hơn rất nhiều cho việc tìm **early signal** của giá cổ phiếu.