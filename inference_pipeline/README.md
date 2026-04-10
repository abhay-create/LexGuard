# LexGuard Offline Inference + Compliance Pipeline

## Setup
```bash
pip install -r requirements_ollama.txt
curl -fsSL https://ollama.com/install.sh | sh
ollama pull llama3
```

## Usage
```bash
python run_pipeline.py --pdf1 ./contract1.pdf --pdf2 ./contract2.pdf
```

With custom arguments:
```bash
python run_pipeline.py --pdf1 ../../Contracts/A.pdf --pdf2 ../../Contracts/B.pdf \
  --model_path ../conreader_new_implementation/exp_test/best_model.pt \
  --out_dir ./results --out_name A_vs_B.json --device cuda --llm_model mistral
```
