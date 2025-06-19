import subprocess
import json, uuid, asyncio
from fastapi import FastAPI, UploadFile, File, Form, HTTPException, WebSocket
from fastapi.responses import FileResponse
from typing import Optional, Union
import os, librosa, numpy as np, torch, pickle, random, math, json
from stream_pipeline_online import StreamSDK
from src.utils import save_temp_file, convert_to_chinese_readable
from fastapi.staticfiles import StaticFiles
from kokoro import KPipeline
import soundfile as sf
from aiortc import RTCPeerConnection, RTCSessionDescription
from aiortc.contrib.media import MediaRelay, MediaPlayer
from starlette.websockets import WebSocketDisconnect
from inference import run  # Assuming inference.py contains the run function


app = FastAPI()
peers = {} 

DATA_ROOT = "./checkpoints/ditto_trt_Ampere_Plus"
CFG_PKL = "./checkpoints/ditto_cfg/v0.4_hubert_cfg_trt_online.pkl"
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

    player = MediaPlayer("static/idle.mp4", format="mp4", loop=True)
    video_sender = pc.addTrack(player.video)        # keep sender handle
    if player.audio:
        pc.addTrack(player.audio)

    await pc.setRemoteDescription(
        RTCSessionDescription(sdp=offer["sdp"], type=offer["type"])
    )
    answer = await pc.createAnswer()
    await pc.setLocalDescription(answer)

    # ── store everything the /speak handler needs ─────────────
    peers[peer_id] = {
        "pc": pc,
        "idle_player": player,
        "video_sender": video_sender,
        "src_img": "static/avatar.png"   # change if you use per-session image
    }

    return {
        "sdp": pc.localDescription.sdp,
        "type": pc.localDescription.type,
        "peer_id": peer_id,
    }

@app.post("/speak")
async def speak(
    peer_id: str = Form(...),
    text: str = Form(...),
    voice_style: str = Form("af_heart")
):
    if peer_id not in peers:
        raise HTTPException(404, "Unknown peer_id")

    entry = peers[peer_id]
    pc = entry["pc"]
    sender = entry["video_sender"]
    src_img = entry["src_img"]

    try:
        # 1) TTS Generation
        wav_path = f"/tmp/{uuid.uuid4().hex}.wav"
        kp = KPipeline(lang_code="a")
        for _, _, audio in kp(text, voice=voice_style):
            sf.write(wav_path, audio, 24000)
            break
        duration = len(audio) / 24000.0

        # 2) Video Generation (continuous, full audio)
        mp4_path = wav_path.replace(".wav", ".mp4")
        tmp_video_path = f"/tmp/{uuid.uuid4().hex}_temp.mp4"

        # Use run_pipeline for correct online-mode batching
        sdk_online.online_mode = True
        await asyncio.to_thread(
            run_pipeline,
            sdk_online,
            wav_path,
            src_img,
            tmp_video_path,
            {
                "setup_kwargs": {},
                "run_kwargs": {"chunksize": (3, 5, 2)}
            }
        )

        # Combine audio and video with ffmpeg
        subprocess.run([
            'ffmpeg', '-loglevel', 'error', '-y',
            '-i', tmp_video_path,
            '-i', wav_path,
            '-map', '0:v',
            '-map', '1:a',
            '-c:v', 'copy',
            '-c:a', 'aac',
            mp4_path
        ], check=True)

        # 3) Create media player
        talk_player = MediaPlayer(mp4_path, format="mp4")
        if not talk_player.video:
            raise HTTPException(500, "Generated video has no video track")

        # Replace video track
        sender.replaceTrack(talk_player.video)
        if talk_player.audio:
            pc.addTrack(talk_player.audio)

        # 4) Cleanup: restore idle after speaking
        async def cleanup():
            await asyncio.sleep(duration + 0.5)
            sender.replaceTrack(entry["idle_player"].video)
            for f in [wav_path, mp4_path, tmp_video_path]:
                try:
                    os.remove(f)
                except Exception:
                    pass

        asyncio.create_task(cleanup())
        return {"ok": True, "length": duration}

    except Exception as e:
        import logging
        logging.exception(f"Error in speak endpoint: {str(e)}")
        raise HTTPException(500, f"Video generation failed: {str(e)}")


@app.post("/offer_offline")
async def offer_offline(offer: dict):
    pc = RTCPeerConnection()
    peer_id = str(uuid.uuid4())

    # Prepare source image and output path
    audio_path = "static/audio.wav"
    src_img = "static/avatar.png"
    mp4_path = f"/tmp/{uuid.uuid4().hex}_offline.mp4"

    # Run offline generation (video only)
    await asyncio.to_thread(
        run,
        sdk,  # assumes preloaded StreamSDK in offline mode
        audio_path,
        src_img,
        mp4_path
    )

    # Create media player from generated video
    player = MediaPlayer(mp4_path, format="mp4")

    video_sender = pc.addTrack(player.video)
    if player.audio:
        pc.addTrack(player.audio)

    await pc.setRemoteDescription(
        RTCSessionDescription(sdp=offer["sdp"], type=offer["type"])
    )
    answer = await pc.createAnswer()
    await pc.setLocalDescription(answer)

    peers[peer_id] = {
        "pc": pc,
        "offline_player": player,
        "video_sender": video_sender,
    }

    return {
        "sdp": pc.localDescription.sdp,
        "type": pc.localDescription.type,
        "peer_id": peer_id,
    }
