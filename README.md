# SLMs in Healthcare: A Survey
Unlike vanilla contextual pre-trained fundamentally _small_ language models (e.g., ClinicalBERT), our interest lies in compressed and optimized approaches for language models in healthcare, developed as a resource-efficient and domain-specialized solution to LLMs.

## Datasets for Healthcare Informatics
1. MIMIC-III, a freely accessible critical care database. [Paper](https://www.nature.com/articles/sdata201635) | [Link](https://github.com/MIT-LCP/mimic-iii-paper/)
2. PubMedQA: A Dataset for Biomedical Research Question Answering. [Paper](https://arxiv.org/abs/1909.06146) | [Link](https://github.com/pubmedqa/pubmedqa)
3. DrugComb: an integrative cancer drug combination data portal. [Paper](https://pubmed.ncbi.nlm.nih.gov/31066443/) | [Link](http://drugcombdb.denglab.org/main)
4. BI55/MedText, a medical diagnosis dataset containing over 1000 top notch textbook quality patient presentations and diagnosis/treatments. [Link](https://huggingface.co/datasets/BI55/MedText)
5. keivalya/MedQuad-MedicalQnADataset: A Question-Entailment Approach to Question Answering. [Paper](https://arxiv.org/abs/1901.08079)
| [Link](https://huggingface.co/datasets/keivalya/MedQuad-MedicalQnADataset)
6. AMEGA-Benchmark: Autonomous medical evaluation for guideline adherence of large language models. [Paper](https://www.nature.com/articles/s41746-024-01356-6) | [Link](https://github.com/DATEXIS/AMEGA-benchmark/tree/main/data)
7. Medical-Diff-VQA: A Large-Scale Medical Dataset for Difference Visual Question Answering on Chest X-Ray Images. [Paper](https://dl.acm.org/doi/abs/10.1145/3580305.3599819) | [Link](https://github.com/Holipori/MIMIC-Diff-VQA)
8. epfl-llm/guidelines: MEDITRON-70B: Scaling Medical Pretraining for Large Language Models. [Paper](https://arxiv.org/abs/2311.16079) | [Link](https://github.com/epfLLM/meditron)
9. MedMCQA : A Large-scale Multi-Subject Multi-Choice Dataset for Medical domain Question Answering. [Paper](https://proceedings.mlr.press/v174/pal22a.html) | [Link](https://github.com/MedMCQA/MedMCQA)

## SLMs for Healthcare
### 100M to 5B parameters
1. BioGPT: generative pre-trained transformer for biomedical text generation and mining. [Paper](https://academic.oup.com/bib/article/23/6/bbac409/6713511?guestAccessKey=a66d9b5d-4f83-4017-bb52-405815c907b9&login=true) | [Model](https://huggingface.co/microsoft/biogpt)
2. BioMedLM: A 2.7B Parameter Language Model Trained On Biomedical Text. [Paper](https://arxiv.org/pdf/2403.18421) | [Model](https://huggingface.co/stanford-crfm/BioMedLM)
3. RadPhi-2: A Vision-Language Foundation Model to Enhance Efficiency of Chest X-ray Interpretation. [Paper](https://arxiv.org/abs/2401.12208) | [Model](https://huggingface.co/StanfordAIMI/RadPhi-2)
4. RadPhi-3: Small Language Models for Radiology. [Paper](https://arxiv.org/abs/2411.13604)
5. TinyLLAMA-1.1B: An Open-Source Small Language Model. [Paper](https://arxiv.org/abs/2401.02385) | [Model](https://github.com/jzhang38/TinyLlama)
6. TinyLlama-1.1B-Medical: A smaller version of https://huggingface.co/therealcyberlord/llama2-qlora-finetuned-medical, which used Llama 2 7B. [Model](https://huggingface.co/therealcyberlord/TinyLlama-1.1B-Medical)
7. Cura-Llama: Evaluating open-source large language Model’s question answering capability on medical domain. [Paper](https://www.ewadirect.com/proceedings/ace/article/view/16000)
8. CancerGPT: for few shot drug pair synergy prediction using large pretrained language models. [Paper](https://www.nature.com/articles/s41746-024-01024-9)
9. ClinicalMamba: A Generative Clinical Language Model on Longitudinal Clinical Notes. [Paper](https://arxiv.org/abs/2403.05795) | [Model](https://github.com/whaleloops/ClinicalMamba)
10. MentalQLM: A lightweight large language model for mental healthcare based on instruction tuning and dual LoRA modules. [Paper](https://www.medrxiv.org/content/10.1101/2024.12.29.24319755v1) | [Model](https://github.com/tortorish/MentalQLM)
11. mhGPT: A Lightweight Generative Pre-Trained Transformer for Mental Health Text Analysis. [Paper](https://arxiv.org/abs/2408.08261)
12. Med-MoE: Mixture of Domain-Specific Experts for Lightweight Medical Vision-Language Models. [Paper](https://arxiv.org/abs/2404.10237) | [Model](https://github.com/jiangsongtao/Med-MoE)
13. HealsHealthAI: Unveiling Personalized Healthcare Insights with Open Source Fine-Tuned LLM. [Paper](https://onlinelibrary.wiley.com/doi/10.1002/9781394249312.ch4)
14. Apollo: A Lightweight Multilingual Medical LLM towards Democratizing Medical AI to 6B People. [Paper](https://arxiv.org/abs/2403.03640) | [Model](https://github.com/FreedomIntelligence/Apollo?tab=readme-ov-file)
15. Med-Pal: Lightweight Large Language Model for Medication Enquiry. [Paper](https://arxiv.org/abs/2407.12822)
### >5B parameters
1. Me-llama: Foundation large language models for medical applications. [Paper](https://arxiv.org/abs/2402.12749)


## Compressed LLMs into SLMs for healthcare


