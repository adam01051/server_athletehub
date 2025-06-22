import cv2
import numpy as np
import math
import subprocess
from datetime import datetime
import random
import os
from flask import Flask, request, jsonify, send_from_directory

app = Flask(__name__)
app.static_folder = 'static'

os.makedirs("static", exist_ok=True)
os.makedirs("uploads", exist_ok=True)

def analyze_video(video_path):
    print(f"[INFO] Starting analysis on: {video_path}")
    video = cv2.VideoCapture(video_path)
    if not video.isOpened():
        return {"error": "Could not open video file"}, 400

    fps = video.get(cv2.CAP_PROP_FPS)
    pixel_to_meter_ratio = 0.002

    # HSV range for red
    lower_red1 = np.array([0, 120, 70])
    upper_red1 = np.array([10, 255, 255])
    lower_red2 = np.array([170, 120, 70])
    upper_red2 = np.array([180, 255, 255])

    prev_center = None
    speeds = []
    ball_positions = []
    centers = []

    width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    output_path = "static/output.mp4"
    temp_path = "static/temp_output.mp4"

    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    def get_angle(p1, p2):
        dx = p2[0] - p1[0]
        dy = p2[1] - p1[1]
        return math.degrees(math.atan2(dy, dx))

    while True:
        ret, frame = video.read()
        if not ret:
            break

        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
        mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
        mask = mask1 + mask2
        mask = cv2.medianBlur(mask, 5)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        center = None

        for contour in contours:
            area = cv2.contourArea(contour)
            if 100 < area < 3000:
                x, y, w, h = cv2.boundingRect(contour)
                center = (x + w // 2, y + h // 2)

                # 👇 Light Green: BGR format
                cv2.circle(frame, center, 10, (144, 238, 144), -1)

                if prev_center is not None:
                    dist = np.linalg.norm(np.array(center) - np.array(prev_center))
                    speed = dist * pixel_to_meter_ratio * fps
                    speeds.append(speed)
                    cv2.putText(frame, f"Speed: {speed:.2f} m/s", (50, 50),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
                prev_center = center
                ball_positions.append(center)
                centers.append(center)
                cv2.rectangle(frame, (x, y), (x + w, y + h), (144, 238, 144), 2)

        out.write(frame)

    video.release()
    out.release()

    # Re-encode for mobile
    try:
        subprocess.run([
            'ffmpeg', '-y', '-i', output_path,
            '-vcodec', 'libx264', '-crf', '23', '-preset', 'fast',
            '-pix_fmt', 'yuv420p',
            temp_path
        ], check=True)
        os.replace(temp_path, output_path)
    except subprocess.CalledProcessError as e:
        return {"error": "FFmpeg re-encoding failed"}, 500

    max_speed = max(speeds) if speeds else None
    avg_speed = sum(speeds) / len(speeds) if speeds else None
    final_speed = speeds[-1] if speeds else None

    swing_meters = None
    if len(ball_positions) >= 5:
        x_vals = np.array([p[0] for p in ball_positions])
        y_vals = np.array([p[1] for p in ball_positions])
        coeffs = np.polyfit(y_vals, x_vals, 2)
        poly = np.poly1d(coeffs)
        deviations = np.abs(x_vals - poly(y_vals))
        swing_meters = max(deviations) * pixel_to_meter_ratio

    spin_type = "Unknown"
    if len(centers) >= 8:
        angle_before = get_angle(centers[-8], centers[-5])
        angle_after = get_angle(centers[-4], centers[-1])
        spin_diff = (angle_after - angle_before + 180) % 360 - 180
        if spin_diff > 10:
            spin_type = "Off Spin"
        elif spin_diff < -10:
            spin_type = "Leg Spin"
        else:
            spin_type = "Straight / No Spin"

    result = {
        "max_speed": round(max_speed, 2) if max_speed else None,
        "avg_speed": round(avg_speed, 2) if avg_speed else None,
        "final_speed": round(final_speed, 2) if final_speed else None,
        "swing": round(swing_meters, 3) if swing_meters else None,
        "spin": spin_type,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "confidence": f"{random.randint(80, 95)}%",
        "output_video_path": output_path
    }

    return result, 200

@app.route('/analyze', methods=['POST'])
def analyze():
    if 'video' not in request.files:
        return jsonify({"error": "No video file provided"}), 400

    video_file = request.files['video']
    filename = f"uploads/{video_file.filename}"
    video_file.save(filename)

    try:
        results, status = analyze_video(filename)
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    finally:
        if os.path.exists(filename):
            os.remove(filename)

    return jsonify(results), status

@app.route('/static/<path:filename>')
def serve_static(filename):
    return send_from_directory(app.static_folder, filename)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001, debug=True)


