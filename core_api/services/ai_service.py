import httpx
import os
import logging
from django.conf import settings

logger = logging.getLogger(__name__)

class AIService:
    """Service to handle AI Inference via Roboflow REST API"""
    
    API_URL = "https://detect.roboflow.com/infer/workflows"
    API_KEY = os.getenv("ROBOFLOW_API_KEY", getattr(settings, 'ROBOFLOW_API_KEY', 'VlMlrQc1hkjaUrctNPdh'))
    WORKSPACE = os.getenv("ROBOFLOW_WORKSPACE", "preciouss-workspace-5hkb5")
    WORKFLOW_ID = os.getenv("ROBOFLOW_WORKFLOW_ID", "detect-count-and-visualize")
    
    @staticmethod
    def analyze_scan(file_path):
        """Send image to Roboflow for anomaly detection"""
        if not AIService.API_KEY or AIService.API_KEY == "YOUR_API_KEY":
            logger.warning("AIService: No Roboflow API Key configured. Skipping AI analysis.")
            return {"success": False, "error": "AI Configuration missing"}

        try:
            url = f"{AIService.API_URL}/{AIService.WORKSPACE}/{AIService.WORKFLOW_ID}?api_key={AIService.API_KEY}"
            
            with httpx.Client(timeout=20.0) as client:
                with open(file_path, "rb") as image_file:
                    files = {"image": image_file} # Workflow uses "image" key
                    response = client.post(url, files=files)
                
                if response.status_code == 200:
                    result = response.json()
                    return AIService.process_workflow_result(result)
                elif response.status_code == 403:
                    logger.error("Roboflow API error: Access Forbidden. Check API Key permissions.")
                    return {"success": False, "error": "AI Service access denied"}
                else:
                    logger.error(f"Roboflow API error: {response.status_code} - {response.text}")
                    return {"success": False, "error": f"AI Inference failed (Status {response.status_code})"}
                    
        except Exception as e:
            logger.warning(f"AIService: Analysis failed due to network/timeout issue: {str(e)}")
            return {"success": False, "error": "AI Service unreachable"}

    @staticmethod
    def process_workflow_result(raw_result):
        """Extract predictions from the Roboflow Workflow output"""
        outputs = raw_result.get("outputs", [])
        if not outputs:
            return {"success": True, "anomalies": [], "count": 0}
            
        # Find the specific node output (usually named after the workflow id)
        node_output = outputs[0].get(AIService.WORKFLOW_ID, {})
        predictions = node_output.get("predictions", [])
        
        anomalies = []
        for pred in predictions:
            # Map workflow coords to 3D space (-10 to 10)
            # Default to 640x480 if image metadata missing in response
            width = node_output.get("image", {}).get("width", 640)
            height = node_output.get("image", {}).get("height", 480)
            
            x_norm = (pred["x"] / width) * 20 - 10
            y_norm = (pred["y"] / height) * 20 - 10
            
            anomalies.append({
                "label": pred["class"],
                "confidence": pred["confidence"],
                "x": round(x_norm, 2),
                "y": round(y_norm, 2),
                "z": 10.0,
                "description": f"Target: {pred['class']} ({round(pred['confidence']*100)}%)"
            })
            
        return {
            "success": True,
            "anomalies": anomalies,
            "count": len(anomalies)
        }
