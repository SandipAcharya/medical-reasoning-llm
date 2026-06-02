# 🚀 Colab Execution Guide (Phase 3)

This guide walks you through exactly how to execute the three notebooks to train and evaluate your Medical Reasoning LLM. 

---

## 📘 Notebook 1: Data Exploration (`01_data_exploration.ipynb`)
**Where to run:** Locally in VSCode OR Google Colab.
**Hardware:** CPU is perfectly fine. No GPU required.

1. Open the notebook in VSCode or Colab.
2. Run all cells from top to bottom.
3. **What to look for:**
   - Watch the dataset download from HuggingFace.
   - Pay attention to **Cell 8**. It will show you the token distribution of the formatted dataset. Ensure that the vast majority of samples fall under 2,048 tokens.
   - Look at the `length_distributions.png` graph generated. 
4. Once this runs successfully, you know the data pipeline is healthy.

---

## 📙 Notebook 2: QLoRA Training (`02_training_colab.ipynb`)
**Where to run:** Google Colab ONLY.
**Hardware:** T4 GPU (Required).

This is the main event. It will take about **4 hours** to complete.

### Step-by-Step Instructions:
1. Go to [Google Colab](https://colab.research.google.com/) and upload `02_training_colab.ipynb` (or push your code to GitHub and open it directly from your repo in Colab).
2. **CRITICAL:** Click `Runtime` -> `Change runtime type` -> select **T4 GPU**.
3. Run **Cell 1** to verify the T4 GPU is attached.
4. Run **Cell 3 (Mount Google Drive)**. A popup will ask for permission to access your Google Drive. **Accept this.** 
   > *Why? Colab deletes all files when it disconnects. Mounting Drive ensures your trained model (`final_adapter`) is permanently saved to your personal Google Drive.*
5. In **Cell 4**, paste your HuggingFace Token into the `HF_TOKEN` variable if you haven't already.
6. Run the rest of the cells. 
7. **During Training (Cell 11):** Keep an eye on the output! Every 100 steps, the `SampleGenerationCallback` will print the model's attempt at answering the heart failure clinical scenario. You will literally watch it get smarter over the 4 hours.
8. **End Result:** When finished, check your Google Drive for a folder called `medical-reasoning-llm/results/final_adapter`. This contains your trained weights!

---

## 📗 Notebook 3: Evaluation (`03_evaluation.ipynb`)
**Where to run:** Google Colab.
**Hardware:** T4 GPU (Required for fast inference).

You run this **after** Notebook 2 has completely finished and saved the adapter to your Drive.

### Step-by-Step Instructions:
1. Open `03_evaluation.ipynb` in Colab. Ensure the Runtime is set to **T4 GPU**.
2. Run the Setup cell. It will mount your Google Drive (so it can access the `final_adapter` you trained in Notebook 2).
3. Ensure the `ADAPTER_DIR` path in **Cell 1** matches where Notebook 2 saved your files in your Drive (usually `/content/drive/MyDrive/medical-reasoning-llm/results/final_adapter`).
4. Run the rest of the cells. 
5. The notebook will merge the base Qwen 3B model with your custom LoRA adapter and run inference on 500 test questions.
6. **End Result:** It will generate an `eval_report.json` and a beautiful `eval_charts.png` showing the ROUGE-L scores and accuracy. 
7. **Action:** Download `eval_charts.png` and `eval_report.json` to your local machine (put them in your local `results/` folder) so you have proof of how well your model performs!

---
## Summary Checklist
- [ ] Run Notebook 1 (Locally/Colab) to verify data.
- [ ] Upload Notebook 2 to Colab, attach T4 GPU, Mount Drive.
- [ ] Let Notebook 2 train for ~4 hours and save `final_adapter` to Drive.
- [ ] Upload Notebook 3 to Colab, attach T4 GPU, Mount Drive.
- [ ] Evaluate the model and download the final charts.
