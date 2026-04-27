# **ExpAIr: Transformer-Based Sentiment Intelligence**

## **Project Overview**
ExpAIr is a specialized sentiment analysis platform leveraging state-of-the-art **DistilBERT** models. It is designed to provide deep linguistic insights through a high-speed, responsive interface.

---

## **Primary Features**

* **Real-time Analysis**: Immediate classification of individual text strings with high-precision confidence scoring.
* **Batch Processing**: Capability to upload and analyze large CSV datasets (up to 20,000+ rows) simultaneously.
* **Low Latency**: Optimized Transformer architecture ensures rapid inference times.
* **Data Visualization**: Automatic generation of sentiment distribution charts and thematic word clouds.

---

## **Technical Specifications**

> **Note**: For batch processing, the system requires a specific file structure to map the data correctly.

| **Requirement** | **Detail** |
| :--- | :--- |
| **Model Engine** | DistilBERT (Transformer-based) |
| **Interface** | Streamlit |
| **File Format** | CSV |
| **Required Header** | Must contain a column named **"text"** |

---

## **Visual Intelligence**

### **Sentiment Metrics**
Upon execution, the system provides a quantitative breakdown of the data:
* **Positive vs. Negative**: Total count and percentage distribution displayed via interactive pie charts.
* **Confidence Intervals**: Statistical certainty for every processed entry.

### **Word Cloud Generation**
The interface generates a visual mapping of the most frequent terms. This allows users to identify recurring themes—such as "flight," "service," or "customer"—at a glance.

---

## **Operational Workflow**

1.  **Selection**: Choose between the **Analysis** or **Batch Processing** tabs.
2.  **Input**: Enter text manually or upload a `.csv` file.
3.  **Configuration**: Use the slider to select the sample size for analysis.
4.  **Execution**: Click **Process Dataset** to trigger the Transformer model.
5.  **Output**: Review the generated visual reports and download the final results.

---

## **Performance Settings**
* **Latency**: Set to **Low** for optimal real-time feedback.
* **Mode**: Configured for **Batch + Realtime** hybrid operations.
