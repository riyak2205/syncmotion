!pip install gradio moviepy librosa opencv-python-headless numpy --quiet
!pip install gradio moviepy librosa opencv-python-headless numpy soundfile --quiet
!pip install protobuf==4.25.3 --upgrade --quiet
!pip install gradio moviepy librosa opencv-python-headless yt-dlp mediapipe --quiet
!pip install mediapipe opencv-python
!pip install gradio mediapipe moviepy opencv-python scipy numpy

import gradio as gr
import numpy as np
import os
import cv2
import mediapipe as mp
from moviepy.editor import ImageSequenceClip, AudioFileClip
from scipy.io.wavfile import write as write_wav

# Preloaded music samples
music_options = {
    "Calm Beat": "music/calm.mp3",
    "Upbeat Tune": "music/upbeat.mp3",
    "Cinematic": "music/cinematic.mp3"
}

# === Motion Generation ===
def generate_motion_frames(image, action, fps=15, duration=4):
    img = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    h, w = img.shape[:2]
    frames = []
    num_frames = int(duration * fps)

    for i in range(num_frames):
        t = i / fps
        warped = img.copy()

        if action == "Jump":
            dy = int(40 * abs(np.sin(np.pi * t)))
            M = np.float32([[1, 0, 0], [0, 1, -dy]])
            warped = cv2.warpAffine(img, M, (w, h), borderMode=cv2.BORDER_REFLECT)

        elif action == "Run":
            dx = int(15 * np.sin(2 * np.pi * t))
            dy = int(5 * abs(np.sin(4 * np.pi * t)))
            M = np.float32([[1, 0, dx], [0, 1, -dy]])
            warped = cv2.warpAffine(img, M, (w, h), borderMode=cv2.BORDER_REFLECT)

        elif action == "Hop":
            dy = int(25 * abs(np.sin(5 * np.pi * t)))
            M = np.float32([[1, 0, 0], [0, 1, -dy]])
            warped = cv2.warpAffine(img, M, (w, h), borderMode=cv2.BORDER_REFLECT)

        elif action == "Slide":
            dx = int(50 * np.sin(np.pi * t))
            M = np.float32([[1, 0, dx], [0, 1, 0]])
            warped = cv2.warpAffine(img, M, (w, h), borderMode=cv2.BORDER_REFLECT)

        elif action == "Pulse":
            scale = 1 + 0.05 * np.sin(2 * np.pi * t)
            M = cv2.getRotationMatrix2D((w // 2, h // 2), 0, scale)
            warped = cv2.warpAffine(img, M, (w, h), borderMode=cv2.BORDER_REPLICATE)

        frames.append(warped)

    return frames

# === Cartoonify Filter ===
def cartoonify(image):
    img = np.array(image)
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    gray = cv2.medianBlur(gray, 5)
    edges = cv2.adaptiveThreshold(gray, 255,
                                  cv2.ADAPTIVE_THRESH_MEAN_C,
                                  cv2.THRESH_BINARY, 9, 9)
    color = cv2.bilateralFilter(img, 9, 300, 300)
    cartoon = cv2.bitwise_and(color, color, mask=edges)
    return cartoon

# === Video Generation ===
def generate_action_video_with_audio(image, audio_data, action, music_choice, cartoon):
    try:
        if cartoon:
            image = cartoonify(image)

        # Handle music source
        if audio_data is None and music_choice in music_options:
            audio_path = music_options[music_choice]
        elif isinstance(audio_data, tuple):
            temp_audio_path = "temp_uploaded_audio.wav"
            rate, data = audio_data
            write_wav(temp_audio_path, rate, data)
            audio_path = temp_audio_path
        else:
            return "No valid audio provided."

        duration = AudioFileClip(audio_path).duration
        frames = generate_motion_frames(image, action, fps=15, duration=duration)
        if not frames:
            return "❌ No frames generated."

        video_clip = ImageSequenceClip([cv2.cvtColor(f, cv2.COLOR_BGR2RGB) for f in frames], fps=15)
        audio_clip = AudioFileClip(audio_path).subclip(0, min(video_clip.duration, duration))
        final_clip = video_clip.set_audio(audio_clip)

        output_video_path = "animated_output.mp4"
        final_clip.write_videofile(output_video_path, codec="libx264", audio_codec="aac", verbose=False)
        return output_video_path

    except Exception as e:
        return f"❌ Error: {e}"

# === CSS ===
custom_css = """
body, .gradio-container {
    background-color: #1e1e2f;
    color: #f1f1f1;
    font-family: 'Poppins', sans-serif;
    padding: 20px;
}
.gr-markdown h1 {
    text-align: center;
    font-size: 3em;
    font-weight: 800;
    color: #ffffff;
    margin-bottom: 30px;
    letter-spacing: 1px;
}
.gr-button {
    background-color: #44475a !important;
    color: #ffffff !important;
    font-weight: 600;
    border-radius: 10px;
    padding: 12px 26px;
    border: 1px solid #6272a4;
    transition: background 0.2s ease-in-out;
}
.gr-button:hover {
    background-color: #6272a4 !important;
    color: #fff !important;
}
.gr-radio input[type="radio"] {
    display: inline-block;
    margin: 0 10px;
}
"""

# === Gradio UI ===
with gr.Blocks(css=custom_css) as demo:
    gr.Markdown("""<h1>🎬 SyncMotion</h1>""")

    with gr.Row():
        image_input = gr.Image(type="pil", label="Upload Photo")
        audio_input = gr.Audio(type="numpy", label="Upload Music (Optional)")

    action_input = gr.Dropdown(
        ["Jump", "Run", "Hop", "Slide", "Pulse"],
        label="Choose an Action"
    )

    with gr.Row():
        music_choice = gr.Dropdown(
            choices=["None"] + list(music_options.keys()),
            label="Or choose background music",
            value="None",
            elem_id="music_select"
        )
        cartoon_toggle = gr.Checkbox(label="Cartoonify My Photo")

    submit_btn = gr.Button("Generate Animation")
    output_video = gr.Video(label="Animated Video", height=360)
    download_btn = gr.File(label="Download Link", visible=False)

    def generate_and_return_video(*args):
        path = generate_action_video_with_audio(*args)
        return path, path if path.endswith(".mp4") else None

    submit_btn.click(
        fn=generate_and_return_video,
        inputs=[image_input, audio_input, action_input, music_choice, cartoon_toggle],
        outputs=[output_video, download_btn]
    )

    gr.Markdown("""
    <br><br>
    <center style='color: #999;'>Built with ❤️ by Riya</center>
    """)

demo.launch(share=True)
