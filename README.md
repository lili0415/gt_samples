# FigEdit: Benchmark for Scientific Figure Editing

[![License](https://img.shields.io/badge/License-Adobe%20Research-red.svg)](./LICENSE.md)
![Data](https://img.shields.io/badge/Dataset-30k%2B_Samples-blue.svg)
![Task](https://img.shields.io/badge/Task-Structured_Editing-green.svg)
![Python](https://img.shields.io/badge/Python-3.x-blue.svg)

## 📖 Introduction

**FigEdit** is a comprehensive benchmark designed to bridge the gap between pixel-level generative models and the structured nature of scientific charts.

Generative models, such as diffusion and autoregressive approaches, have demonstrated impressive capabilities in editing natural images. However, applying these tools to scientific charts rests on a flawed assumption: a chart is not merely an arrangement of pixels but a visual representation of structured data governed by a graphical grammar. Consequently, chart editing is not a pixel-manipulation task but a **structured transformation problem**.

To address this fundamental mismatch, we introduce **FigEdit**, a large-scale benchmark for scientific figure editing comprising over 30,000 samples. Grounded in real-world data, our benchmark is distinguished by its diversity, covering 10 distinct chart types and a rich vocabulary of complex editing instructions.

The benchmark is organized into five distinct and progressively challenging tasks:
1.  **Single Edits**: Atomic, one-step operations.
2.  **Multi Edits**: Complex instructions requiring sequential operations.
3.  **Conversational Edits**: Iterative refinement via dialogue.
4.  **Visual Guidance**: Edits driven by visual cues (Visual Circle/Tag-to-Image).
5.  **Style Transfer**: Applying aesthetic styles from a reference chart.

By releasing FigEdit, we aim to enable systematic progress in structure-aware figure editing and provide a common ground for fair comparison.

## 📊 Dataset Structure

The dataset supports five core tasks, distributed across specific JSON files:

| Task | Description | Source File | Mode Filter |
| :--- | :--- | :--- | :--- |
| **Single** | Single-turn instruction editing. | `instructions.json` | `mode="single"` |
| **Multi** | Multi-turn/Multi-step instruction editing. | `instructions.json` | `mode="multi"` |
| **Conv** | Conversational/Dialog-based editing. | `annotation_combo2.json` | N/A (Conversational structure) |
| **Visual** | Tag-to-Image / Visual Circle consistency. | `instructions_visual_circle_from_tags.json` | `mode="single_visual_circle_from_tags"` |
| **Transfer** | Reference-based Style Transfer. | `instructions_style_transfer_single.json` | `mode="style_transfer_single"` |

## 🧪 Evaluation

We provide a unified evaluation script (`eval_unified.py`) that calculates both low-level vision metrics and high-level semantic scores.

### Metrics
The evaluation pipeline computes the following metrics:
* **Vision Metrics**: SSIM, PSNR, LPIPS.
* **Semantic Consistency**: CLIP Similarity (Image-to-Image).
* **Text Accuracy**: OCR Similarity (Text content preservation/update).
* **LLM Judge**: A GPT-4o based judge that evaluates:
    * *Instruction Following*: Did the model follow the edit command?
    * *Content Preservation*: Is the original data preserved where not edited?
    * *Image Quality*: General visual fidelity.

### Prerequisites

Ensure you have the necessary dependencies installed. If you intend to use the **LLM Judge**, you must set the following environment variables for Azure OpenAI:

```bash
export AZURE_OPENAI_API_KEY="your_key_here"
export AZURE_OPENAI_ENDPOINT="[https://your-endpoint.openai.azure.com/](https://your-endpoint.openai.azure.com/)"
export AZURE_OPENAI_API_VERSION="2024-12-01-preview"
export AZURE_OPENAI_DEPLOYMENT="gpt-4o-mini-20240718"
