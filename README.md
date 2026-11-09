# Skull Auto-Label

Upload a skull video. The app extracts frames, runs the YOLO-seg model (`Skull_31_10_69.pt`, 19 classes) on every frame, and returns a zip in YOLO segmentation format with a `data.yaml`. Upload that zip to Roboflow and correct the masks there.

Run locally: `pip install -r requirements.txt && streamlit run app.py`

Set the password in Streamlit Cloud under **Settings → Secrets**: `APP_PASSWORD = "..."`. The default is `123456`.
