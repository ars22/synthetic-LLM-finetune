# synthetic-LLM-finetune

Using this repository we aim to run experiments that test the fundamental limits of using synthetic data for aligning LLMs. 
- Is SFT enough?
- When is RLHF (oneline DPO) using synthetic data strictly better?
- When can we do better than SFT data?
- What role does model size have to play here?

- conda create -n syth-llm python==3.10
- conda activate syth-llm
- pip install -r requirements.txt
- FLASH_ATTENTION_SKIP_CUDA_BUILD=TRUE pip install flash-attn --no-build-isolation