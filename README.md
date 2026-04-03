# Vehicle Counter

A web application that counts vehicles in uploaded videos using **YOLOv8m** for object detection and **BotSort** for multi-object tracking, ensuring each vehicle is counted only once.

---

## Features

- Upload any traffic/road video directly from your browser
- Detects and counts multiple vehicle types (cars, trucks, buses, motorcycles, etc.)
- Uses **BotSort** tracker to assign unique IDs to each vehicle, preventing duplicate counts
- Powered by the **YOLOv8m** model for accurate, real-time detection
- Displays the total vehicle count after processing

---

## Tech Stack

| Layer | Technology |
|-------|-----------|
| Object Detection | [YOLOv8m](https://docs.ultralytics.com/) (Ultralytics) |
| Object Tracking | [BotSort](https://github.com/NirAharon/BoT-SORT) |
| Backend | Python |
| Frontend | Web interface (video upload form) |

---

## How It Works

1. **Upload** a video file through the web interface.
2. The backend processes each frame using **YOLOv8m** to detect vehicles.
3. **BotSort** assigns a unique ID to every detected vehicle and maintains that ID across frames, minimizing duplicate counts even when a vehicle briefly leaves and re-enters the frame.
4. Once the video is fully processed, the application returns the **total unique vehicle count**.

---

## Getting Started

### Prerequisites

- Python 3.8+
- pip

### Installation

```bash
# Clone the repository
git clone https://github.com/sarthak-cc/Vehicle-Counter.git
cd Vehicle-Counter

# Install dependencies
pip install -r requirements.txt
```

### Running the Application

```bash
python app.py
```

Open your browser and navigate to `http://localhost:5000` (or the port shown in the terminal).

---

## Usage

1. Open the web app in your browser.
2. Click **Upload Video** and select a road/traffic video.
3. Click **Submit** and wait for the video to be processed.
4. The total vehicle count will be displayed on the results page.

---

## Model Details

- **YOLOv8m** – a medium-sized variant of the YOLOv8 architecture, balancing speed and accuracy for vehicle detection.
- **BotSort** – a robust multi-object tracker that combines appearance and motion cues to maintain consistent object identities across video frames, eliminating duplicate counts.

---

## License

This project is licensed under the [MIT License](LICENSE). Feel free to use, modify, and distribute it.
