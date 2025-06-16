import subprocess
import json, uuid, asyncio
from fastapi import FastAPI, UploadFile, File, Form, HTTPException, WebSocket
from fastapi.responses import FileResponse
from typing import Optional, Union
import os, librosa, numpy as np, torch, pickle, random, math, json
from stream_pipeline_offline import StreamSDK
from src.utils import save_temp_file, convert_to_chinese_readable
from fastapi.staticfiles import StaticFiles
from kokoro import KPipeline
import soundfile as sf
from aiortc import RTCPeerConnection, RTCSessionDescription
from aiortc.contrib.media import MediaRelay, MediaPlayer
from starlette.websockets import WebSocketDisconnect


app = FastAPI()
peers = {} 

DATA_ROOT = "./checkpoints/ditto_trt_Ampere_Plus"
CFG_PKL = "./checkpoints/ditto_cfg/v0.4_hubert_cfg_trt.pkl"
sdk = StreamSDK(CFG_PKL, DATA_ROOT)

sdk_online = StreamSDK(CFG_PKL, DATA_ROOT)
sdk_online.online_mode = True

def run_pipeline(SDK, audio_path, source_path, output_path, more_kwargs):
    setup_kwargs = more_kwargs.get("setup_kwargs", {})
    run_kwargs = more_kwargs.get("run_kwargs", {})

    SDK.setup(source_path, output_path, **setup_kwargs)
    audio, _ = librosa.core.load(audio_path, sr=16000)
    num_f = math.ceil(len(audio) / 16000 * 25)

    SDK.setup_Nd(
        N_d=num_f,
        fade_in=run_kwargs.get("fade_in", -1),
        fade_out=run_kwargs.get("fade_out", -1),
        ctrl_info=run_kwargs.get("ctrl_info", {})
    )

    if SDK.online_mode:
        chunksize = run_kwargs.get("chunksize", (3, 5, 2))
        audio = np.concatenate([np.zeros((chunksize[0] * 640,), dtype=np.float32), audio], 0)
        split_len = int(sum(chunksize) * 0.04 * 16000) + 80
        for i in range(0, len(audio), chunksize[1] * 640):
            chunk = audio[i:i + split_len]
            if len(chunk) < split_len:
                chunk = np.pad(chunk, (0, split_len - len(chunk)), mode="constant")
            SDK.run_chunk(chunk, chunksize)
    else:
        feat = SDK.wav2feat.wav2feat(audio)
        SDK.audio2motion_queue.put(feat)

    SDK.close()
    os.system(f'ffmpeg -loglevel error -y -i "{SDK.tmp_output_path}" -i "{audio_path}" -map 0:v -map 1:a -c:v copy -c:a aac "{output_path}"')
    return output_path


@app.post("/generate-video")
async def generate_and_download(
    audio_file: UploadFile = File(...),
    source_file: UploadFile = File(...),
    more_kwargs: str = Form(default="{}")
):
    audio_path = save_temp_file(audio_file)
    source_path = save_temp_file(source_file)
    output_path = audio_path.replace(".wav", "_output.mp4")

    try:
        kwargs_dict = json.loads(more_kwargs)
    except json.JSONDecodeError:
        raise HTTPException(status_code=400, detail="Invalid JSON in 'more_kwargs'.")

    output_file = run_pipeline(
        SDK=sdk,
        audio_path=audio_path,
        source_path=source_path,
        output_path=output_path,
        more_kwargs=kwargs_dict
    )

    if not os.path.isfile(output_file):
        raise HTTPException(status_code=500, detail="Output file was not created.")

    return FileResponse(
        path=output_file,
        media_type="video/mp4",
        filename=os.path.basename(output_file)
    )


@app.post("/video-voice-clone")
async def generate_video_from_text(
    source_file: UploadFile = File(...),
    input_text: str = Form(...),
    more_kwargs: str = Form(default="{}"),
    clone_audio: Optional[UploadFile] = File(None),
    speed: Optional[Union[float, str]] = Form(None),
    voice_style: Optional[str] = Form(None),
    language: Optional[str] = Form("a")  # default: American English
):
    source_path = save_temp_file(source_file)
    audio_out = f"/tmp/{uuid.uuid4().hex}.wav"
    video_out = audio_out.replace(".wav", "_output.mp4")

    try:
        kwargs_dict = json.loads(more_kwargs)
    except json.JSONDecodeError:
        raise HTTPException(status_code=400, detail="Invalid JSON in 'more_kwargs'.")

    ref_audio_path = None
    if clone_audio:
        if language.lower().startswith("z"):
            input_text = convert_to_chinese_readable(input_text)
        ref_audio_path = save_temp_file(clone_audio)
        tts_cmd = [
            "f5-tts_infer-cli",
            "--model", "F5TTS_v1_Base",
            "--gen_text", input_text,
            "--output_file", audio_out,
            "--ref_audio", ref_audio_path,
            "--ref_text", ""
        ]
        if speed:
            tts_cmd += ["--speed", str(speed)]
        subprocess.run([str(arg) for arg in tts_cmd], check=True)
        if not os.path.isfile(audio_out):
            raise HTTPException(status_code=500, detail="TTS output file not found.")

    else:
        if not voice_style:
            raise HTTPException(status_code=400, detail="voice_style required without clone_audio")
        pipeline = KPipeline(lang_code=language)
        generator = pipeline(
            input_text,
            voice=voice_style,
            speed=float(speed) if speed else 1.0,
            split_pattern=r"\n+"
        )
        for _, _, audio in generator:
            sf.write(audio_out, audio, 24000)
            break
        if not os.path.isfile(audio_out):
            raise HTTPException(status_code=500, detail="KokoroTTS generation failed.")

    final_video = run_pipeline(sdk, audio_out, source_path, video_out, kwargs_dict)
    if not os.path.isfile(final_video):
        raise HTTPException(status_code=500, detail="Video generation failed.")

    return FileResponse(
        path=final_video,
        media_type="video/mp4",
        filename=os.path.basename(final_video)
    )
    
''' WebSocket handling for real-time video generation can be added here if needed.'''
### API Goes here ###

app.mount("/static", StaticFiles(directory="static"), name="static")
relay = MediaRelay()

@app.post("/offer")
async def webrtc_offer(offer: dict):
    pc = RTCPeerConnection()
    peer_id = str(uuid.uuid4())
    peers[peer_id] = pc

    await pc.setRemoteDescription(
        RTCSessionDescription(sdp=offer["sdp"], type=offer["type"])
    )

    player = MediaPlayer("static/idle.mp4", format="mp4", loop=True)
    print(f"MediaPlayer created for session {peer_id} - video: {player.video}, audio: {player.audio}")

    if player.video:
        pc.addTrack(player.video)
    else:
        raise Exception("No video track available in media file")

    answer = await pc.createAnswer()
    await pc.setLocalDescription(answer)

    return {
        "sdp": pc.localDescription.sdp,
        "type": pc.localDescription.type,
        "peer_id": peer_id
    }