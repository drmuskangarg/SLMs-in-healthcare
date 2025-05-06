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
10. ColonINST: Frontiers in intelligent colonoscopy [Paper](https://arxiv.org/abs/2410.17241) | [Link](https://github.com/ai4colonoscopy/IntelliScope)

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
16. ColonGPT: Frontiers in intelligent colonoscopy [Paper](https://arxiv.org/abs/2410.17241) | [Model](https://github.com/ai4colonoscopy/IntelliScope)

### >5B parameters
1. Me-llama: Foundation large language models for medical applications. [Paper](https://arxiv.org/abs/2402.12749)
2. BioMistral: A Collection of Open-Source Pretrained Large Language Models for Medical Domains. [Paper](https://arxiv.org/abs/2402.10373)
3. BioMistral-NLU: Towards More Generalizable Medical Language Understanding through Instruction Tuning. [Paper](https://arxiv.org/abs/2410.18955) | [Model](https://github.com/uw-bionlp/BioMistral-NLU)
4. Meerkat-7B: Small Language Models Learn Enhanced Reasoning Skills from Medical Textbooks. [Paper](https://arxiv.org/abs/2404.00376) | [Model](https://huggingface.co/dmis-lab/meerkat-7b-v1.0)
5. MentaLLaMA: Interpretable Mental Health Analysis on Social Media with Large Language Models. [Paper](https://arxiv.org/abs/2309.13567) | [Model](https://github.com/SteveKGYang/MentalLLaMA)
6. Med-R^2: Crafting Trustworthy LLM Physicians through Retrieval and Reasoning of Evidence-Based Medicine. [Paper](https://arxiv.org/abs/2501.11885) | [Model](https://github.com/8023looker/Med-RR)
7. ChatDoctor: A Medical Chat Model Fine-Tuned on a Large Language Model Meta-AI (LLaMA) Using Medical Domain Knowledge. [Paper](https://arxiv.org/abs/2303.14070) | [Model](https://github.com/Kent0n-Li/ChatDoctor)
8. MEDITRON-7B: Scaling Medical Pretraining for Large Language Models. [Paper](https://arxiv.org/abs/2311.16079) | [Model](https://github.com/epfLLM/meditron)
9. PMC-LLaMA: Towards Building Open-source Language Models for Medicine. [Paper](https://arxiv.org/abs/2304.14454) | [Model](https://github.com/chaoyi-wu/PMC-LLaMA)
10. MedAlpaca: An Open-Source Collection of Medical Conversational AI Models and Training Data. [Paper](https://arxiv.org/abs/2304.08247) | [Model](https://github.com/kbressem/medAlpaca)
11. AlpaCare:Instruction-tuned Large Language Models for Medical Application. [Paper](https://arxiv.org/abs/2310.14558) | [Model](https://github.com/XZhang97666/AlpaCare)
12. Psy-LLM: Scaling up Global Mental Health Psychological Services with AI-based Large Language Models. [Paper](https://arxiv.org/abs/2307.11991)
13. CancerLLM: A Large Language Model in Cancer Domain. [Paper](https://arxiv.org/abs/2406.10459)


## Optimize SLMs for healthcare
### Pretraining strategies
1. Large language model benchmarks in medical tasks. [Paper](https://arxiv.org/pdf/2410.21348)
2. Llama-3-Meditron: An Open-Weight Suite of Medical LLMs Based on Llama-3.1 [Paper](https://openreview.net/forum?id=ZcD35zKujO)
3. Apollo: A Lightweight Multilingual Medical LLM towards Democratizing Medical AI to 6B People. [Paper](https://arxiv.org/abs/2403.03640)
4. Large Language Models and Large Multimodal Models in Medical Imaging: A Primer for Physicians. [Paper](https://jnm.snmjournals.org/content/66/2/173.abstract)
5. Large language models are poor medical coders—benchmarking of medical code querying. [Paper](https://ai.nejm.org/doi/full/10.1056/AIdbp2300040)
6. Leveraging large language models for clinical abbreviation disambiguation. [Paper](https://ai.nejm.org/doi/full/10.1056/AIdbp2300040)

### Attention mechanisms
1. Attention is not all you need: the complicated case of ethically using large language models in healthcare and medicine. [Paper](https://www.thelancet.com/journals/ebiom/article/PIIS2352-3964(23)00077-4/fulltext?ref=dedataverbinders.nl)

### Prompt Engineering
1. Pre-train, prompt, and predict: A systematic survey of prompting methods in natural language processing. [Paper](https://dl.acm.org/doi/full/10.1145/3560815)
2. Prompt engineering for healthcare: Methodologies and applications. [Paper](https://arxiv.org/abs/2304.14670)
3. Chain of thought utilization in large language models and application in nephrology. [Paper](https://www.mdpi.com/1648-9144/60/1/148)
4. ATSCOT: Chain of Thought for Structuring Anesthesia Medical Records. [Paper](https://ieeexplore.ieee.org/abstract/document/10776558)
5. CoMT: Chain-of-medical-thought reduces hallucination in medical report generation. [Paper](https://ieeexplore.ieee.org/abstract/document/10887699)
6. Surgraw: Multi-agent workflow with chain-of-thought reasoning for surgical intelligence. [Paper](https://arxiv.org/abs/2503.10265)
7. AutoMedPrompt: A New Framework for Optimizing LLM Medical Prompts Using Textual Gradients. [Paper](https://arxiv.org/abs/2502.15944)

### Fine-tuning
1. Do we still need clinical language models? [Paper](https://proceedings.mlr.press/v209/eric23a)
2. Health Care Language Models and Their Fine-Tuning for Information Extraction: Scoping Review. [Paper]([https://proceedings.mlr.press/v209/eric23a](https://medinform.jmir.org/2024/1/e60164/))
3. How to Design, Create, and Evaluate an Instruction-Tuning Dataset for Large Language Model Training in Health Care: Tutorial From a Clinical Perspective. [Paper](https://www.jmir.org/2025/1/e70481/)
4. BioInstruct: instruction tuning of large language models for biomedical natural language processing. [Paper](https://academic.oup.com/jamia/article/31/9/1821/7687618)
5. Instruction Tuning Large Language Models to Understand Electronic Health Records. [Paper](https://openreview.net/forum?id=Dgy5WVgPd2#discussion)
6. Medalign: A clinician-generated dataset for instruction following with electronic medical records. [Paper](https://ojs.aaai.org/index.php/AAAI/article/view/30205)
7. LlamaCare: An Instruction Fine-Tuned Large Language Model for Clinical NLP. [Paper](https://aclanthology.org/2024.lrec-main.930/)
8. Alpacare: Instruction-tuned large language models for medical application. [Paper](https://arxiv.org/abs/2310.14558)
9. LEAP: LLM instruction-example adaptive prompting framework for biomedical relation extraction. [Paper](https://academic.oup.com/jamia/article/31/9/2010/7696965)
10. Mdagents: An adaptive collaboration of llms for medical decision-making. [Paper](https://proceedings.neurips.cc/paper_files/paper/2024/hash/90d1fc07f46e31387978b88e7e057a31-Abstract-Conference.html)

### Knowledge distillation
1. Distilling the knowledge from large-language model for health event prediction. [Paper](https://www.nature.com/articles/s41598-024-75331-2)
2. Large language model distilling medication recommendation model. [Paper](https://arxiv.org/abs/2402.02803)
3. SleepCoT: A Lightweight Personalized Sleep Health Model via Chain-of-Thought Distillation. [Paper](https://arxiv.org/abs/2410.16924)
4. Non-small cell lung cancer detection through knowledge distillation approach with teaching assistant. [Paper](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0306441)
5. Distilling Large Language Models for Efficient Clinical Information Extraction. [Paper](https://arxiv.org/abs/2501.00031)
6. LLM-Enhanced Multi-Teacher Knowledge Distillation for Modality-Incomplete Emotion Recognition in Daily Healthcare. [Paper](https://ieeexplore.ieee.org/document/10697478)
7. Adapt-and-Distill: Developing Small, Fast and Effective Pretrained Language Models for Domains. [Paper](https://arxiv.org/abs/2106.13474)

### Quantization
1. Privacy-Preserving SAM Quantization for Efficient Edge Intelligence in Healthcare. [Paper](https://arxiv.org/abs/2410.01813)
2. Mental Healthcare Chatbot Based on Custom Diagnosis Documents Using a Quantized Large Language Model. [Paper](https://ieeexplore.ieee.org/document/10522371)
3. MentalQLM: A lightweight large language model for mental healthcare based on instruction tuning and dual LoRA modules. [Paper](https://www.medrxiv.org/content/10.1101/2024.12.29.24319755v1)
4. BioMedLM: A 2.7B Parameter Language Model Trained On Biomedical Text. [Paper](https://arxiv.org/abs/2403.18421)
5. mhGPT: A Lightweight Generative Pre-Trained Transformer for Mental Health Text Analysis. [Paper](https://arxiv.org/abs/2408.08261)
6. BioMistral: A Collection of Open-Source Pretrained Large Language Models for Medical Domains. [Paper](https://arxiv.org/abs/2402.10373)

### Pruning
1. All-in-One Tuning and Structural Pruning for Domain-Specific LLMs. [Paper](https://arxiv.org/abs/2412.14426)
2. Pruning as a Domain-specific LLM Extractor. [Paper](https://arxiv.org/abs/2405.06275)
3. LLM-Pruner: On the Structural Pruning of Large Language Models. [Paper](https://arxiv.org/abs/2305.11627)

### Reasoning
1. Chain of thought utilization in large language models and application in nephrology. [Paper](https://www.mdpi.com/1648-9144/60/1/148)
2. ATSCOT: Chain of Thought for Structuring Anesthesia Medical Records. [Paper](https://ieeexplore.ieee.org/abstract/document/10776558)
3. CoMT: Chain-of-medical-thought reduces hallucination in medical report generation. [Paper](https://ieeexplore.ieee.org/abstract/document/10887699)
4. Surgraw: Multi-agent workflow with chain-of-thought reasoning for surgical intelligence. [Paper](https://arxiv.org/abs/2503.10265)
5. AutoMedPrompt: A New Framework for Optimizing LLM Medical Prompts Using Textual Gradients. [Paper](https://arxiv.org/abs/2502.15944)
6. Merging Clinical Knowledge into Large Language Models for Medical Research and Applications: A Survey. [Paper](https://arxiv.org/abs/2502.20988)
7. AlzheimerRAG: Multimodal Retrieval Augmented Generation for PubMed articles. [Paper](https://arxiv.org/abs/2412.16701)
8. RT: a Retrieving and Chain-of-Thought framework for few-shot medical named entity recognition. [Paper](https://academic.oup.com/jamia/article/31/9/1929/7665312)
9. Integrating Chain-of-Thought and Retrieval Augmented Generation Enhances Rare Disease Diagnosis from Clinical Notes. [Paper](https://arxiv.org/abs/2503.12286)


