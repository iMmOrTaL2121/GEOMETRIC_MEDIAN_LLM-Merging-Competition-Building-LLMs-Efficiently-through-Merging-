<div align="center">
  <h1>LLM-Merging: Building LLMs Efficiently through Merging</h1>

  [![](https://img.shields.io/badge/Documentation-online-green)](https://llm-merging.readthedocs.io)
  [![](https://img.shields.io/badge/Website-online-green)](https://llm-merging.github.io)
  [![](https://img.shields.io/badge/License-MIT-blue)](#License)
</div>

This repository contains our submission for the LLM-Merging competition.

Training high-performing large language models (LLMs) from scratch is a notoriously expensive and challenging task. The **LLM Merging Competition (NeurIPS 2024 Challenge): Building LLMs Efficiently through Merging** promotes research in model merging techniques, where pretrained LLMs are fine-tuned for specific tasks and then combined to generate LLMs that can perform well across a wide variety of skills, such as reasoning, coding, math, chat, and tool use. Our submission to the competition introduces an approach that associates each pretrained LLM with a "task vector" relative to a “Base LLM”. These “task vectors” are derived from the LoRA (Low Rank Adaptation) weights of pretrained LLM's. We compute the **geometric median** of these task vectors in a high-dimensional space, applying **Weiszfeld’s iterative algorithm** and adding it to **pretrained base LLM**, effectively merging the models to generalize their capabilities and achieve state-of-the-art results on benchmark tests.
