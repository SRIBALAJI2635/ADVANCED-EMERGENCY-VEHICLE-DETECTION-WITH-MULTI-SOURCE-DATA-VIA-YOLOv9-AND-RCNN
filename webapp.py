import argparse
import io
import os
import cv2
import numpy as np
import subprocess
from flask import Flask, render_template, request, Response, send_from_directory
from PIL import Image
from ultralytics import YOLO

app = Flask(__name__)

# Ensure the uploads directory exists
UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

@app.route("/")
def hello_world():
    return render_template('index.html')

@app.route("/", methods=["GET", "POST"])
def predict_img():
    if request.method == "POST":
        if 'file' in request.files:
            f = request.files['file']
            filepath = os.path.join(UPLOAD_FOLDER, f.filename)
            f.save(filepath)

            # Determine the file extension
            file_extension = f.filename.rsplit('.', 1)[1].lower()

            if file_extension in ['jpg', 'jpeg', 'png']:  # Added 'jpeg' and 'png' support
                img = cv2.imread(filepath)
                model = YOLO('yoloooo.pt')
                detections = model(img)  # Run detection
                result_path = f"runs/detect/{f.filename}"
                cv2.imwrite(result_path, detections[0].plot())  # Save the result image
                return display(f.filename)

            elif file_extension == 'mp4':
                return process_video(filepath)

    return render_template('index.html')

def process_video(video_path):
    cap = cv2.VideoCapture(video_path)
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    output_path = 'output.mp4'
    out = cv2.VideoWriter(output_path, fourcc, 30.0, (frame_width, frame_height))
    model = YOLO('yoloooo.pt')

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        results = model(frame)
        res_plotted = results[0].plot()  # Add detections to the frame
        out.write(res_plotted)  # Save to output video

    cap.release()
    out.release()
    return video_feed()

@app.route('/uploads/<filename>')
def display(filename):
    return send_from_directory(UPLOAD_FOLDER, filename)

def get_frame():
    video_path = 'output.mp4'
    video = cv2.VideoCapture(video_path)
    while True:
        success, image = video.read()
        if not success:
            break
        ret, jpeg = cv2.imencode('.jpg', image)
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + jpeg.tobytes() + b'\r\n\r\n')

@app.route("/video_feed")
def video_feed():
    return Response(get_frame(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route("/webcam_feed")
def webcam_feed():
    cap = cv2.VideoCapture(0)

    def generate():
        while True:
            success, frame = cap.read()
            if not success:
                break
            img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            model = YOLO('yoloooo.pt')
            results = model(img)  # Run detection
            res_plotted = results[0].plot()
            frame = cv2.cvtColor(res_plotted, cv2.COLOR_RGB2BGR)
            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()

            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')

    return Response(generate(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route("/rtmp_feed")
def rtmp_feed():
    rtmp_url = 'rtmp://a.rtmp.youtube.com/live2/ued9-7z7b-k65z-f1ux-0yj8'
    ffmpeg_cmd = [
        'ffmpeg',
        '-i', rtmp_url,
        '-f', 'rawvideo',
        '-pix_fmt', 'bgr24',
        '-an', '-sn', '-vcodec', 'rawvideo',
        'pipe:1'
    ]

    process = subprocess.Popen(ffmpeg_cmd, stdout=subprocess.PIPE)

    def generate():
        while True:
            raw_frame = process.stdout.read(640 * 480 * 3)  # Adjust for your video resolution
            if not raw_frame:
                break
            
            frame = np.frombuffer(raw_frame, np.uint8).reshape((480, 640, 3))  # Adjust for your video resolution
            img = Image.fromarray(frame)
            model = YOLO('yoloooo.pt')
            results = model(img)
            res_plotted = results[0].plot()
            img_BGR = cv2.cvtColor(res_plotted, cv2.COLOR_RGB2BGR)
            frame = cv2.imencode('.jpg', img_BGR)[1].tobytes()

            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')

    return Response(generate(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Flask app exposing YOLOv9 models")
    parser.add_argument("--port", default=5000, type=int, help="port number")
    args = parser.parse_args()
    app.run(host="0.0.0.0", port=args.port, debug=True)
