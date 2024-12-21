from flask import Flask, request, jsonify
from .video_processor import VideoProcessor
from .density_analyzer import DensityAnalyzer
from .heatmap_generator import HeatmapGenerator
import cv2
import numpy as np
from datetime import datetime

app = Flask(__name__)

@app.route('/analyze', methods=['POST'])
def analyze_video():
    try:
        data = request.get_json()
        
        # Validate input
        if not data.get('video_stream_url'):
            return jsonify({'error': 'video_stream_url is required'}), 400
            
        # Initialize processors
        video_processor = VideoProcessor()
        density_analyzer = DensityAnalyzer()
        heatmap_generator = HeatmapGenerator()
        
        # Process video stream
        footfall_data = video_processor.process_stream(
            data['video_stream_url'],
            data.get('zones', [])
        )
        
        # Analyze density
        density_data = density_analyzer.analyze(footfall_data)
        
        # Generate heatmap
        heatmap_urls = heatmap_generator.generate(density_data)
        
        response = {
            "footfall_summary": {
                "total_footfall": footfall_data['total_count'],
                "zone_footfall": footfall_data['zone_counts'],
                "high_density_times": density_data['high_density_periods']
            },
            "heatmap_urls": heatmap_urls
        }
        
        return jsonify(response)
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000) 
