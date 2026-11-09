"""Skull bone auto-labeler.

Upload a video -> frames are extracted -> the YOLO-seg model labels every frame ->
download a zip (YOLO segmentation format + data.yaml) ready to upload to Roboflow.
"""
import hmac
import json
import tempfile
import time
import zipfile
from pathlib import Path

import cv2
import streamlit as st

MODEL_PATH = Path(__file__).parent / "Skull_31_10_69.pt"
DEFAULT_IMGSZ = 1280  # the model was trained at 1280

st.set_page_config(page_title="Skull Auto-Label", page_icon="💀", layout="centered")


# ------------------------------------------------------------------ password
def check_password() -> bool:
    expected = str(st.secrets.get("APP_PASSWORD", "123456"))
    if st.session_state.get("authed"):
        return True
    st.title("💀 Skull Auto-Label")
    pw = st.text_input("รหัสผ่าน", type="password")
    if st.button("เข้าสู่ระบบ"):
        if hmac.compare_digest(pw, expected):
            st.session_state["authed"] = True
            st.rerun()
        else:
            st.error("รหัสผ่านไม่ถูกต้อง")
    return False


if not check_password():
    st.stop()


# ------------------------------------------------------------------ model
@st.cache_resource(show_spinner="กำลังโหลดโมเดล AI ...")
def load_model():
    from ultralytics import YOLO
    return YOLO(str(MODEL_PATH))


def label_lines(result, conf_min: float) -> list[str]:
    """YOLO-seg polygon lines (class x1 y1 x2 y2 ...) for one prediction."""
    if result.masks is None or len(result.masks) == 0:
        return []
    lines = []
    for box, polygon in zip(result.boxes, result.masks.xyn):
        if float(box.conf[0]) < conf_min or len(polygon) < 3:
            continue
        coords = " ".join(f"{x:.6f} {y:.6f}" for x, y in polygon)
        lines.append(f"{int(box.cls[0])} {coords}")
    return lines


def process_video(video_path: Path, zip_path: Path, prefix: str, fps: float,
                  conf_min: float, imgsz: int, progress, status) -> dict:
    model = load_model()
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError("เปิดไฟล์วิดีโอไม่ได้ (ไฟล์เสียหรือไม่รองรับ codec นี้)")

    src_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
    step = max(1, round(src_fps / fps))
    n_target = max(1, -(-total // step)) if total else 0

    n_frames = n_labeled = idx = 0
    started = time.time()
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if idx % step == 0:
                name = f"{prefix}_frame_{n_frames:05d}"
                ok_jpg, jpg = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
                if ok_jpg:
                    zf.writestr(f"{name}.jpg", jpg.tobytes())
                    res = model.predict(frame, conf=0.1, imgsz=imgsz,
                                        retina_masks=True, verbose=False)[0]
                    lines = label_lines(res, conf_min)
                    if lines:
                        zf.writestr(f"{name}.txt", "\n".join(lines))
                        n_labeled += 1
                    n_frames += 1
                    if n_target:
                        progress.progress(min(n_frames / n_target, 1.0))
                    elapsed = time.time() - started
                    eta = ""
                    if n_target and n_frames:
                        remain = elapsed / n_frames * (n_target - n_frames)
                        eta = f" · เหลือประมาณ {remain / 60:.1f} นาที"
                    status.write(f"ประมวลผลแล้ว {n_frames}/{n_target or '?'} เฟรม{eta}")
            idx += 1
        cap.release()

        names = model.names
        yaml = ["names:"]
        yaml += [f"  {i}: {json.dumps(names[i])}" for i in sorted(names)]
        yaml.append(f"nc: {len(names)}")
        zf.writestr("data.yaml", "\n".join(yaml) + "\n")

    return {"frames": n_frames, "labeled": n_labeled,
            "minutes": (time.time() - started) / 60}


# ------------------------------------------------------------------ UI
st.title("💀 Skull Auto-Label")
st.write(
    "อัปโหลดวิดีโอ → AI ตัดเฟรมและวาด mask กระดูกให้ทุกเฟรม → "
    "ดาวน์โหลดไฟล์ zip ไปอัปโหลดเข้า **Roboflow** แล้วแก้ไขต่อที่นั่น"
)

video = st.file_uploader("1️⃣ เลือกไฟล์วิดีโอ", type=["mp4", "mov", "avi", "mkv", "m4v"])

with st.expander("⚙️ ตั้งค่า (ปกติไม่ต้องแก้)"):
    fps = st.number_input("จำนวนเฟรมที่ตัดต่อวินาทีของวิดีโอ", 0.1, 30.0, 1.0, 0.5)
    conf_min = st.slider("ความมั่นใจขั้นต่ำของ AI (ต่ำ = ได้ mask มากขึ้น)", 0.05, 0.9, 0.3, 0.05)
    imgsz = st.select_slider("ความละเอียดตอนให้ AI ดู (ต่ำ = เร็วขึ้น)",
                             options=[640, 960, 1280], value=DEFAULT_IMGSZ)

if video is not None and st.button("2️⃣ เริ่มให้ AI label", type="primary"):
    work = Path(tempfile.mkdtemp(prefix="skull_"))
    video_path = work / f"input{Path(video.name).suffix or '.mp4'}"
    video_path.write_bytes(video.getbuffer())
    stem = "".join(c if c.isalnum() else "_" for c in Path(video.name).stem)[:40]
    prefix = f"{stem}_{time.strftime('%Y%m%d_%H%M%S')}"
    zip_path = work / f"roboflow_{prefix}.zip"

    st.info("⏳ กำลังประมวลผล ห้ามปิดหน้านี้ — วิดีโอยาวอาจใช้เวลาหลายนาที")
    progress = st.progress(0.0)
    status = st.empty()
    try:
        info = process_video(video_path, zip_path, prefix, fps, conf_min, imgsz, progress, status)
    except Exception as e:  # show the doctor a readable error instead of a traceback
        st.error(f"เกิดข้อผิดพลาด: {e}")
        st.stop()
    finally:
        video_path.unlink(missing_ok=True)
    st.session_state["result"] = {"zip": str(zip_path), "name": zip_path.name, **info}

result = st.session_state.get("result")
if result and Path(result["zip"]).exists():
    st.success(
        f"✅ เสร็จแล้ว: {result['frames']} เฟรม · AI วาด mask ได้ {result['labeled']} เฟรม "
        f"· ใช้เวลา {result['minutes']:.1f} นาที"
    )
    with open(result["zip"], "rb") as f:
        st.download_button("3️⃣ ดาวน์โหลดไฟล์ zip สำหรับ Roboflow", f,
                           file_name=result["name"], mime="application/zip", type="primary")
    st.caption("ใน Roboflow: เปิดโปรเจกต์ → Upload Data → ลากไฟล์ zip นี้ลงไป (ไม่ต้องแตกไฟล์)")
