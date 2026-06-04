import os
# Force HuggingFace to download to D: drive since C: is out of space
os.environ["HF_HOME"] = os.path.join(os.getcwd(), "hf_cache")
from dotenv import load_dotenv
load_dotenv()
os.environ["HF_TOKEN"] = os.getenv("HF_TOKEN", "")

import logging
from flask import Flask, render_template, request, jsonify
from src.medical_reasoning.inference.pipeline import MedicalReasoningPipeline

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# Global variable for the pipeline
pipeline = None

def init_pipeline():
    global pipeline
    if pipeline is None:
        try:
            logger.info("Initializing Medical Reasoning Pipeline...")
            # Automatically detect if running on CPU or GPU
            import torch
            load_in_4bit = torch.cuda.is_available()
            
            pipeline = MedicalReasoningPipeline.from_pretrained(
                base_model="Qwen/Qwen2.5-3B-Instruct",
                adapter_path="./results/final_adapter",
                load_in_4bit=True,
                device_map={"model": 0, "lm_head": 0},
            )
            logger.info("Pipeline initialized successfully.")
        except Exception as e:
            import traceback
            logger.error(f"Failed to initialize pipeline: {e}")
            logger.error(traceback.format_exc())

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/api/reason", methods=["POST"])
def reason():
    global pipeline
    if pipeline is None:
        init_pipeline()
        
    if not pipeline:
        return jsonify({"error": "Model pipeline is not initialized or adapter not found."}), 500
        
    data = request.json
    question = data.get("question")
    
    if not question:
        return jsonify({"error": "Question is required."}), 400
        
    try:
        result = pipeline.reason(question)
        return jsonify({
            "reasoning_chain": result.reasoning_chain,
            "final_answer": result.final_answer,
            "generation_time_s": result.generation_time_s,
            "num_tokens_generated": result.num_tokens_generated,
            "confidence_note": result.confidence_note
        })
    except Exception as e:
        logger.error(f"Inference error: {e}")
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    logger.info("Starting Flask server. Pipeline will load on first request.")
    app.run(host="0.0.0.0", port=5000, debug=True, use_reloader=False)
