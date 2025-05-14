# GeneRAG: A Retrieval-Augmented Generation System for Gene Function Analysis

GeneRAG is a specialized Retrieval-Augmented Generation (RAG) system designed for researchers and bioinformaticians who need to analyze gene functions and relationships in scientific literature. By combining powerful language models with a curated vector database of scientific publications, GeneRAG provides detailed, citation-backed information about genes and their functions in various contexts.

## Features

- **Scientific Accuracy**: Retrieves information from peer-reviewed scientific literature with proper citations
- **Comprehensive Analysis**: Provides detailed insights on gene functions, relationships, and roles in diseases
- **Citation Support**: All responses include references to original scientific publications
- **Efficient Knowledge Retrieval**: Uses vector databases for fast, relevant information retrieval
- **User-Friendly API**: Simple interface for querying gene information

## Installation

### From PyPI (recommended)

```bash
pip install generag
```

### From GitHub

```bash
git clone https://github.com/Zaoqu-Liu/GeneRAG
cd GeneRAG
pip install -e .
```

## GeneRAG: A Retrieval-Augmented Generation System for Gene Function Analysis


```python
from generag import GeneRAG, Config
import requests
from IPython.display import Markdown, display

# Global Configuration
API_KEY = "sk-94e47466066143a7a63f010653806ec3"
DB_DIR = "/Users/liuzaoqu/Desktop/OmicSeek/GeneReview_XHx2C/vector_db"
CITATION_DIR = "/Users/liuzaoqu/Desktop/OmicSeek/GeneReview_XHx2C/cite"

config = Config(
    api_key=API_KEY,
    db_dir=DB_DIR,
    citation_dir=CITATION_DIR
)
```


```python
# Create a GeneRAG instance
rag = GeneRAG(config=config)

# Define the query
query = "What is THBS2 in colorectal cancer?"

response = rag.process_query(query)

display(Markdown(response))
```


## **THBS2 in Colorectal Cancer (CRC): Oncogenic Role and Clinical Significance**  

**Thrombospondin-2 (THBS2)** is a matricellular glycoprotein involved in extracellular matrix (ECM) remodeling, angiogenesis, and cancer progression. In colorectal cancer (CRC), THBS2 has emerged as a critical player with **pro-tumorigenic effects**, influencing proliferation, invasion, metastasis, and prognosis.  

### **Key Findings on THBS2 in CRC**  

1. **Overexpression and Poor Prognosis**  
   - THBS2 is significantly upregulated in CRC tissues and cell lines compared to normal controls [5][6][9].  
   - High THBS2 expression correlates with **shorter overall survival (OS)** and **disease-free survival (DFS)**, suggesting its role as an independent prognostic marker [6][9][12].  

2. **Promotion of Tumor Aggressiveness**  
   - **In vitro and in vivo studies** demonstrate that THBS2 overexpression enhances CRC cell **proliferation, migration, and invasion**, while its knockdown suppresses these effects [5][12].  
   - THBS2 activates the **Wnt/β-catenin signaling pathway**, a key driver of CRC progression [5].  

3. **Association with Metastasis and EMT**  
   - THBS2, along with **BGN (Biglycan)**, promotes **epithelial-mesenchymal transition (EMT)** in CRC, increasing metastatic potential [12].  
   - Knockdown of THBS2 reduces expression of EMT markers (Snail, Vimentin, N-cadherin) and sensitizes CRC cells to chemotherapy (5-FU, irinotecan, oxaliplatin) [12].  

4. **Immune Microenvironment Modulation**  
   - THBS2 expression positively correlates with **immune cell infiltration** in CRC, suggesting a role in shaping the tumor microenvironment [6].  
   - It is linked to **PI3K-Akt signaling** and ECM organization, further supporting its involvement in tumor-stroma interactions [6].  

5. **Diagnostic and Therapeutic Potential**  
   - THBS2 may serve as a **biomarker** for early CRC detection and prognosis prediction [8][9].  
   - Targeting THBS2 or its regulatory axis (e.g., **miR-584-5p/THBS2**) could offer novel therapeutic strategies [2][5].  

### **Contradictory Roles in Other Cancers**  
While THBS2 acts as an **oncogene in CRC**, it exhibits **tumor-suppressive effects** in gastric cancer (GC) and ovarian cancer, where its downregulation is associated with poor outcomes [3][23]. This highlights its **context-dependent role** in cancer biology.  

### **Conclusion**  
THBS2 is a **critical regulator of CRC progression**, influencing tumor growth, metastasis, and immune evasion. Its overexpression correlates with **aggressive disease and poor survival**, making it a promising **prognostic marker and therapeutic target** in CRC.  

---

### **References**  
[2] Circ_0079530 promotes NSCLC progression via miR-584-5p/THBS2 axis - https://pubmed.ncbi.nlm.nih.gov/36382967  
[3] THBS2 inhibits proliferation and angiogenesis in gastric cancer - https://pubmed.ncbi.nlm.nih.gov/25262009  
[5] THBS2 promotes CRC via Wnt/β-catenin signaling - https://pubmed.ncbi.nlm.nih.gov/35315209  
[6] THBS2 correlates with poor prognosis and immune infiltration in CRC - https://pubmed.ncbi.nlm.nih.gov/34471634  
[9] THBS2 as a prognostic marker in CRC - https://pubmed.ncbi.nlm.nih.gov/30570855  
[12] THBS2/BGN drive EMT and chemoresistance in CRC - https://pubmed.ncbi.nlm.nih.gov/36377223  
[23] THBS2 methylation in ovarian cancer - https://pubmed.ncbi.nlm.nih.gov/18955754



```python
# Perform DeepSeek query
ds_response = requests.post(
    "https://api.deepseek.com/v1/chat/completions",
    headers={"Authorization": f"Bearer {API_KEY}"},
    json={"model": "deepseek-chat", "messages": [{"role": "user", "content": query}]}
).json()["choices"][0]["message"]["content"]

display(Markdown(ds_response))
```


**THBS2 (Thrombospondin-2)** is an extracellular matrix glycoprotein involved in cell adhesion, angiogenesis, and tumor progression. In **colorectal cancer (CRC)**, THBS2 has emerged as a potential biomarker and therapeutic target due to its roles in tumorigenesis and metastasis. Here’s a breakdown of its significance:

### **1. Role in Colorectal Cancer:**
- **Overexpression in CRC:** THBS2 is frequently upregulated in colorectal cancer tissues compared to normal mucosa, particularly in advanced stages and metastatic disease.
- **Pro-Tumorigenic Effects:**  
  - Promotes **tumor growth, invasion, and metastasis** by modulating extracellular matrix (ECM) remodeling.  
  - Enhances **angiogenesis** (though Thrombospondins are generally anti-angiogenic, THBS2 may have context-dependent roles).  
  - Interacts with **TGF-β signaling**, contributing to epithelial-mesenchymal transition (EMT) and cancer cell migration.  
- **Immune Modulation:** May influence the tumor microenvironment by regulating immune cell infiltration.

### **2. Diagnostic and Prognostic Biomarker:**
- **Early Detection:** THBS2 shows promise as a **non-invasive biomarker** for CRC when detected in blood or stool, potentially aiding early diagnosis.  
- **Poor Prognosis:** High THBS2 expression correlates with **worse survival**, lymph node metastasis, and chemoresistance.  

### **3. Therapeutic Implications:**
- **Targeting THBS2** (e.g., via antibodies or inhibitors) could disrupt tumor-stroma interactions and metastasis.  
- **Combination Therapy:** May enhance efficacy of existing treatments like chemotherapy or immunotherapy.  

### **4. Research Highlights:**
- Studies suggest THBS2 promotes CRC progression via **Wnt/β-catenin** and **PI3K/AKT pathways**.  
- Its detection in **liquid biopsies** (e.g., serum or stool DNA tests) is being explored for clinical use.  

### **Conclusion:**  
THBS2 is a multifunctional protein in CRC, driving tumor aggressiveness and serving as a potential biomarker for early diagnosis and prognosis. Further research is needed to validate its clinical utility and therapeutic targeting.  

Would you like details on specific studies or mechanisms?



```python
response = rag.process_query("The relationships of TGFbeta and THBS2?")
display(Markdown(response))
```


## The Relationship Between TGFβ and THBS2  

Thrombospondin-2 (THBS2) is a matricellular glycoprotein involved in extracellular matrix (ECM) remodeling, angiogenesis, and fibrosis. Its interaction with **transforming growth factor-beta (TGFβ)** is critical in multiple pathological and physiological processes, including cancer progression, fibrosis, and inflammation. Below is a synthesis of key findings on their relationship:  

### **1. TGFβ Upregulates THBS2 in Fibrosis and Cancer**  
- In **non-alcoholic fatty liver disease (NAFLD)**, TGFβ activation in hepatic stellate cells (HSCs) increases **THBS2** expression independently of the canonical **TGFβ-SMAD2/3 pathway**, promoting collagen deposition and fibrosis [5].  
- THBS2 knockdown reduces collagen expression, suggesting its role as a downstream effector of TGFβ-driven fibrogenesis [5].  
- In **colorectal cancer (CRC)**, THBS2 overexpression correlates with poor prognosis and activates the **Wnt/β-catenin pathway**, which is often modulated by TGFβ signaling [16].  

### **2. THBS2 as a Mediator of TGFβ’s Pro-Tumorigenic Effects**  
- THBS2 is overexpressed in **non-small cell lung cancer (NSCLC)** and **pancreatic ductal adenocarcinoma (PDAC)**, where it enhances tumor progression and metastasis [9, 17].  
- In **intrahepatic cholangiocarcinoma (iCCA)**, THBS2 (along with THBS1) promotes tumor cell invasion and mesenchymal transition, likely synergizing with TGFβ-induced EMT [20].  
- THBS2’s pro-metastatic effects in **uveal melanoma (UM)** involve **PI3K/AKT pathway** activation, a pathway also regulated by TGFβ [8, 18].  

### **3. THBS2 in TGFβ-Associated Angiogenesis and ECM Remodeling**  
- THBS2 regulates **MMP2 clearance**, and its loss-of-function leads to ECM abnormalities, mimicking TGFβ-driven ECM dysregulation [6].  
- In **gastric cancer (GC)**, THBS2 suppresses angiogenesis by inhibiting endothelial cell proliferation, countering TGFβ’s pro-angiogenic effects in some contexts [12, 24].  

### **4. Clinical Implications**  
- **Diagnostic Biomarker**: Serum THBS2, combined with CA19-9, detects PDAC with **87% sensitivity and 98% specificity** [3, 17].  
- **Therapeutic Target**: THBS2 inhibition may mitigate fibrosis in NAFLD [5] or block metastasis in CRC and NSCLC [9, 16].  

### **Conclusion**  
THBS2 is a multifunctional effector of TGFβ signaling, influencing fibrosis, cancer progression, and angiogenesis. Its context-dependent roles (pro- or anti-tumorigenic) highlight its potential as a biomarker and therapeutic target in TGFβ-associated diseases.  

---  
### **References**  
[3] THBS2 as a biomarker for PDAC - https://pubmed.ncbi.nlm.nih.gov/pubmed/28739842  
[5] TGFβ-THBS2 axis in NAFLD fibrosis - https://pubmed.ncbi.nlm.nih.gov/pubmed/38010940  
[6] THBS2 and ECM regulation - https://pubmed.ncbi.nlm.nih.gov/pubmed/38433265  
[8] THBS2 in UM metastasis - https://pubmed.ncbi.nlm.nih.gov/pubmed/30338785  
[9] THBS2 in NSCLC diagnosis - https://pubmed.ncbi.nlm.nih.gov/pubmed/37658862  
[12] THBS2’s anti-angiogenic role in GC - https://pubmed.ncbi.nlm.nih.gov/pubmed/25262009  
[16] THBS2-Wnt/β-catenin in CRC - https://pubmed.ncbi.nlm.nih.gov/pubmed/35315209  
[17] THBS2-CA19-9 panel for PDAC - https://pubmed.ncbi.nlm.nih.gov/pubmed/28701476  
[20] THBS2 in iCCA progression - https://pubmed.ncbi.nlm.nih.gov/pubmed/38339060  
[24] THBS2 in GC prognosis - https://pubmed.ncbi.nlm.nih.gov/pubmed/25262009
