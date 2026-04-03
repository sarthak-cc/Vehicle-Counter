import os
import threading
import time
import uuid
from collections import defaultdict

import cv2
from flask import (
	Flask,
	jsonify,
	render_template,
	request,
	send_file,
	abort,
)
from ultralytics import YOLO
from werkzeug.utils import secure_filename


# Global configuration
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_FOLDER = os.path.join(BASE_DIR, "uploads")
OUTPUT_FOLDER = os.path.join(BASE_DIR, "outputs")
MAX_FILE_SIZE = 100 * 1024 * 1024  # 100MB
ALLOWED_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv"}


# Ensure folders exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)


# Load YOLO model
model = YOLO(os.path.join(BASE_DIR, "yolov8n.pt"))


# Jobs dictionary to track background processing
jobs = {}


app = Flask(__name__, template_folder="templates")


def allowed_file_extension(filename: str) -> bool:
	_, ext = os.path.splitext(filename)
	return ext.lower() in ALLOWED_EXTENSIONS


def get_file_size(file_storage) -> int:
	"""Safely determine uploaded file size in bytes."""
	pos = file_storage.stream.tell()
	file_storage.stream.seek(0, os.SEEK_END)
	size = file_storage.stream.tell()
	file_storage.stream.seek(pos, os.SEEK_SET)
	return size


def remove_file_later(path: str, delay: int = 60) -> None:
	"""Remove a file after a delay in a background thread."""

	def _remove():
		try:
			time.sleep(delay)
			if os.path.exists(path):
				os.remove(path)
		except Exception:
			# Best-effort cleanup; ignore errors
			pass

	t = threading.Thread(target=_remove, daemon=True)
	t.start()


def process_video(job_id: str, input_path: str, output_path: str, line_pos: float, conf: float) -> None:
	"""Background video processing for vehicle counting using YOLOv8m."""

	try:
		cap = cv2.VideoCapture(input_path)
		if not cap.isOpened():
			if job_id in jobs:
				jobs[job_id]["status"] = "error"
				jobs[job_id]["error"] = "Unable to open video file."
			return

		width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
		height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
		W = width
		H = height
		PROCESS_WIDTH = 640
		scale = PROCESS_WIDTH / W
		PROCESS_H = int(H * scale)
		fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
		total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0

		line_y = int(height * line_pos)

		fourcc = cv2.VideoWriter_fourcc(*"mp4v")
		writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

		track_history = defaultdict(list)
		counted_ids = set()
		last_centers_y = {}

		vehicle_count = 0
		class_names = {2: "Car", 3: "Motorbike", 5: "Bus", 7: "Truck"}
		class_counts = {2: 0, 3: 0, 5: 0, 7: 0}

		frame_num = 0

		while True:
			ret, frame = cap.read()
			if not ret:
				break

			frame_num += 1

			small_frame = cv2.resize(frame, (PROCESS_WIDTH, PROCESS_H))

			results = model.track(
				small_frame,
				persist=True,
				tracker="botsort.yaml",
				classes=[2, 3, 5, 7],
				conf=conf,
				verbose=False,
			)

			# Draw counting line
			cv2.line(
				frame,
				(0, line_y),
				(width, line_y),
				(0, 255, 255),
				2,
			)

			if results and len(results) > 0:
				r = results[0]
				boxes = getattr(r, "boxes", None)

				if boxes is not None and hasattr(boxes, "xyxy"):
					for box in boxes:
						xyxy = box.xyxy[0].tolist()
						x1, y1, x2, y2 = xyxy
						x1 = int(x1 / scale)
						y1 = int(y1 / scale)
						x2 = int(x2 / scale)
						y2 = int(y2 / scale)

						cls_id = int(box.cls[0]) if box.cls is not None else -1
						track_id = int(box.id[0]) if box.id is not None else None
						conf_score = float(box.conf[0]) if box.conf is not None else 0.0

						# Compute center
						cx = int((x1 + x2) / 2)
						cy = int((y1 + y2) / 2)

						if track_id is not None:
							track_history[track_id].append((cx, cy))

						# Draw bounding box
						cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

						# Label text
						label_cls = class_names.get(cls_id, str(cls_id))
						label = f"{label_cls} {track_id if track_id is not None else ''} {conf_score:.2f}"
						cv2.putText(
							frame,
							label,
							(x1, max(0, y1 - 10)),
							cv2.FONT_HERSHEY_SIMPLEX,
							0.5,
							(0, 255, 0),
							2,
							cv2.LINE_AA,
						)

						# Draw center dot
						cv2.circle(frame, (cx, cy), 4, (0, 0, 255), -1)

						# Draw trajectory trail
						if track_id is not None:
							pts = track_history[track_id]
							if len(pts) > 1:
								for i in range(1, len(pts)):
									cv2.line(
										frame,
										pts[i - 1],
										pts[i],
										(255, 0, 0),
										2,
									)

						# Counting logic: crossing the line from top to bottom
						if track_id is not None and cls_id in class_counts:
							prev_y = last_centers_y.get(track_id)
							last_centers_y[track_id] = cy

							if (
								prev_y is not None
								and prev_y < line_y <= cy
								and track_id not in counted_ids
							):
								counted_ids.add(track_id)
								vehicle_count += 1
								class_counts[cls_id] += 1

			# Stats overlay (semi-transparent black)
			overlay = frame.copy()
			cv2.rectangle(overlay, (0, 0), (380, 130), (0, 0, 0), -1)
			frame = cv2.addWeighted(overlay, 0.5, frame, 0.5, 0)

			stats_lines = [
				f"Total: {vehicle_count}",
				f"Cars: {class_counts[2]}",
				f"Bikes: {class_counts[3]}",
				f"Buses: {class_counts[5]}",
				f"Trucks: {class_counts[7]}",
			]

			y_offset = 25
			for text in stats_lines:
				cv2.putText(
					frame,
					text,
					(10, y_offset),
					cv2.FONT_HERSHEY_SIMPLEX,
					0.6,
					(255, 255, 255),
					2,
					cv2.LINE_AA,
				)
				y_offset += 25

			if frame_num % 10 == 0:
				frame_path = os.path.join(OUTPUT_FOLDER, f"frame_{job_id}.jpg")
				cv2.imwrite(frame_path, frame)
				if job_id in jobs:
					jobs[job_id]["current_frame_path"] = frame_path

			writer.write(frame)

			if job_id in jobs and total_frames > 0:
				progress = int(frame_num / total_frames * 100)
				jobs[job_id]["progress"] = max(0, min(progress, 100))

		cap.release()
		writer.release()

		frame_path = f"outputs/frame_{job_id}.jpg"
		if os.path.exists(frame_path):
			os.remove(frame_path)

		if job_id in jobs:
			jobs[job_id].update(
				{
					"status": "done",
					"progress": 100,
					"total": vehicle_count,
					"cars": class_counts[2],
					"bikes": class_counts[3],
					"buses": class_counts[5],
					"trucks": class_counts[7],
					"output_filename": os.path.basename(output_path),
				}
			)

	except Exception as e:
		if job_id in jobs:
			jobs[job_id]["status"] = "error"
			jobs[job_id]["error"] = str(e)
	finally:
		# Clean up input file after processing
		try:
			if os.path.exists(input_path):
				os.remove(input_path)
		except Exception:
			pass


@app.route("/", methods=["GET"])
def index():
	return render_template("index.html")


@app.route("/upload", methods=["POST"])
def upload():
	if "file" not in request.files:
		return jsonify({"error": "No file part in the request."}), 400

	file = request.files.get("file")

	if file is None or file.filename == "":
		return jsonify({"error": "No file selected for uploading."}), 400

	if not allowed_file_extension(file.filename):
		return (
			jsonify(
				{
					"error": "Invalid file type. Allowed: .mp4, .avi, .mov, .mkv.",
				}
			),
			400,
		)

	size = get_file_size(file)
	if size > MAX_FILE_SIZE:
		return (
			jsonify(
				{
					"error": "File too large. Maximum allowed size is 100MB.",
				}
			),
			400,
		)

	filename = secure_filename(file.filename)
	input_path = os.path.join(UPLOAD_FOLDER, filename)
	file.save(input_path)

	cap_check = cv2.VideoCapture(input_path)
	fps = cap_check.get(cv2.CAP_PROP_FPS)
	frames = cap_check.get(cv2.CAP_PROP_FRAME_COUNT)
	duration = frames / fps
	cap_check.release()
	if duration > 120:
		os.remove(input_path)
		return jsonify({"error": "Video too long. Please upload under 2 minutes."}), 400

	# Line position (fraction of height)
	line_pos_raw = request.form.get("line_position", "0.55")
	try:
		line_pos = float(line_pos_raw)
	except ValueError:
		line_pos = 0.55

	# Sensitivity mapping
	sensitivity = request.form.get("sensitivity", "medium").lower()
	if sensitivity == "low":
		conf = 0.25
	elif sensitivity == "high":
		conf = 0.50
	else:  # medium or anything else
		conf = 0.35

	job_id = str(uuid.uuid4())
	jobs[job_id] = {"status": "processing", "progress": 0}

	output_filename = f"{job_id}_output.mp4"
	output_path = os.path.join(OUTPUT_FOLDER, output_filename)

	t = threading.Thread(
		target=process_video,
		args=(job_id, input_path, output_path, line_pos, conf),
		daemon=True,
	)
	t.start()

	return jsonify({"job_id": job_id}), 200


@app.route("/sample")
def sample():
	import urllib.request

	sample_path = os.path.join(UPLOAD_FOLDER, "sample_traffic.mp4")

	# Download sample video if it does not exist
	if not os.path.exists(sample_path):
		try:
			url = "https://www.pexels.com/download/video/2103099/?fps=25.0&h=360&w=640"
			urllib.request.urlretrieve(url, sample_path)
		except Exception as e:
			return jsonify({"error": "Could not download sample video. Please upload your own."}), 500

	# Generate job id
	job_id = str(uuid.uuid4())
	output_filename = f"output_{job_id}.mp4"
	output_path = os.path.join(OUTPUT_FOLDER, output_filename)

	# Store job
	jobs[job_id] = {"status": "processing", "progress": 0}

	# Run in background thread
	thread = threading.Thread(
		target=process_video,
		args=(job_id, sample_path, output_path, 0.55, 0.35)
	)
	thread.daemon = True
	thread.start()

	return jsonify({"job_id": job_id})


@app.route("/status/<job_id>", methods=["GET"])
def status(job_id: str):
	if job_id not in jobs:
		return jsonify({"error": "Job ID not found."}), 404
	return jsonify(jobs[job_id])


@app.route("/current-frame/<job_id>", methods=["GET"])
def current_frame(job_id: str):
	if job_id not in jobs:
		return jsonify({"error": "Job ID not found."}), 404

	if "current_frame_path" not in jobs[job_id]:
		return jsonify({"error": "Current frame not found."}), 404

	frame_path = jobs[job_id]["current_frame_path"]
	if not os.path.exists(frame_path):
		return jsonify({"error": "Current frame not found."}), 404

	return send_file(frame_path, mimetype="image/jpeg")


@app.route("/download/<filename>")
def download(filename):
	file_path = os.path.join(OUTPUT_FOLDER, filename)
	if not os.path.exists(file_path):
		return jsonify({"error": "File not found"}), 404
	return send_file(
		file_path,
		as_attachment=True,
		download_name=filename
	)


@app.route("/stream/<filename>")
def stream(filename):
	file_path = os.path.join(OUTPUT_FOLDER, filename)
	if not os.path.exists(file_path):
		return jsonify({"error": "File not found"}), 404

	response = send_file(
		file_path,
		mimetype="video/mp4",
		conditional=True
	)
	response.headers["Accept-Ranges"] = "bytes"
	response.headers["Cache-Control"] = "no-cache"
	return response


@app.route("/preview", methods=["POST"])
def preview():
	if "file" not in request.files:
		return jsonify({"error": "No file part in the request."}), 400

	file = request.files.get("file")

	if file is None or file.filename == "":
		return jsonify({"error": "No file selected for uploading."}), 400

	if not allowed_file_extension(file.filename):
		return (
			jsonify(
				{
					"error": "Invalid file type. Allowed: .mp4, .avi, .mov, .mkv.",
				}
			),
			400,
		)

	filename = f"preview_{uuid.uuid4().hex}_" + secure_filename(file.filename)
	temp_path = os.path.join(UPLOAD_FOLDER, filename)
	file.save(temp_path)

	try:
		cap = cv2.VideoCapture(temp_path)
		if not cap.isOpened():
			return jsonify({"error": "Unable to open video for preview."}), 400

		# Try to grab frame 30; if not available, fallback to first frame
		target_frame = 30
		total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
		if total_frames > target_frame:
			cap.set(cv2.CAP_PROP_POS_FRAMES, target_frame)

		ret, frame = cap.read()
		if not ret:
			cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
			ret, frame = cap.read()

		cap.release()

		if not ret or frame is None:
			return jsonify({"error": "Unable to read frame from video."}), 400

		success, buffer = cv2.imencode(".jpg", frame)
		if not success:
			return jsonify({"error": "Failed to encode preview frame."}), 500

		from flask import Response

		return Response(buffer.tobytes(), mimetype="image/jpeg")

	finally:
		try:
			if os.path.exists(temp_path):
				os.remove(temp_path)
		except Exception:
			pass


if __name__ == "__main__":
	app.run(host="0.0.0.0", port=7860, debug=False)
