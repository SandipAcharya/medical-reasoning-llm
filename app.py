import os
import json
import time
# Force HuggingFace to download to D: drive since C: is out of space
os.environ["HF_HOME"] = os.path.join(os.getcwd(), "hf_cache")
from dotenv import load_dotenv
load_dotenv()
os.environ["HF_TOKEN"] = os.getenv("HF_TOKEN", "")

import logging
from flask import Flask, render_template, request, jsonify, Response, stream_with_context
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

@app.route("/api/reason/stream", methods=["POST"])
def reason_stream():
    """Server-Sent Events endpoint — streams tokens to the browser as they are generated."""
    global pipeline
    if pipeline is None:
        init_pipeline()
    if not pipeline:
        return jsonify({"error": "Model pipeline is not initialized."}), 500

    data = request.json
    question = data.get("question")
    if not question:
        return jsonify({"error": "Question is required."}), 400

    def generate():
        t0 = time.time()
        full_text = ""
        try:
            for chunk in pipeline.reason_stream(question):
                full_text += chunk
                # Send each token chunk as an SSE event
                payload = json.dumps({"token": chunk, "done": False})
                yield f"data: {payload}\n\n"
        except Exception as e:
            logger.error(f"Streaming error: {e}")
            yield f"data: {json.dumps({'error': str(e), 'done': True})}\n\n"
            return

        elapsed = round(time.time() - t0, 2)
        # Send final event with parsed reasoning + answer
        reasoning = pipeline._extract_reasoning(full_text)
        answer = pipeline._extract_answer(full_text)
        final_payload = json.dumps({
            "done": True,
            "reasoning_chain": reasoning,
            "final_answer": answer,
            "generation_time_s": elapsed,
        })
        yield f"data: {final_payload}\n\n"

    return Response(
        stream_with_context(generate()),
        mimetype="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        }
    )

if __name__ == "__main__":
    logger.info("Starting Flask server. Pipeline will load on first request.")
    app.run(host="0.0.0.0", port=5000, debug=False, use_reloader=False, threaded=True)

