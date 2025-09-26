"use client";

import { LoadingSVG } from "@/components/button/LoadingSVG";
import { ChatMessageType } from "@/components/chat/ChatTile";
import { ColorPicker } from "@/components/colorPicker/ColorPicker";
import { AudioInputTile } from "@/components/config/AudioInputTile";
import { ConfigurationPanelItem } from "@/components/config/ConfigurationPanelItem";
import { NameValueRow } from "@/components/config/NameValueRow";
import { PlaygroundHeader } from "@/components/playground/PlaygroundHeader";
import {
  PlaygroundTab,
  PlaygroundTabbedTile,
  PlaygroundTile,
} from "@/components/playground/PlaygroundTile";
import { useConfig } from "@/hooks/useConfig";
import { TranscriptionTile } from "@/transcriptions/TranscriptionTile";
import {
  BarVisualizer,
  VideoTrack,
  useConnectionState,
  useDataChannel,
  useLocalParticipant,
  useRoomInfo,
  useTracks,
  useVoiceAssistant,
  useRoomContext,
  useParticipantAttributes,
} from "@livekit/components-react";
import { ConnectionState, LocalParticipant, Track, RoomEvent } from "livekit-client";
import type { TrackReference } from "@livekit/components-react";
import { QRCodeSVG } from "qrcode.react";
import { ReactNode, useCallback, useEffect, useMemo, useState, useRef } from "react";
import tailwindTheme from "../../lib/tailwindTheme.preval";
import { EditableNameValueRow } from "@/components/config/NameValueRow";
import { AttributesInspector } from "@/components/config/AttributesInspector";
import { RpcPanel } from "./RpcPanel";
import { ChatMessageInput } from "@/components/chat/ChatMessageInput";

export const API_URL = "";

/* -------------------- New: voice map (label → code) -------------------- */
const VOICE_OPTIONS: { label: string; value: string }[] = [
  { label: "American Female Alloy", value: "af_alloy" },
  { label: "American Female Aoede", value: "af_aoede" },
  { label: "American Female Bella", value: "af_bella" },
  { label: "American Female Jessica", value: "af_jessica" },
  { label: "American Female Kore", value: "af_kore" },
  { label: "American Female Nicole", value: "af_nicole" },
  { label: "American Female Nova", value: "af_nova" },
  { label: "American Female River", value: "af_river" },
  { label: "American Female Sarah", value: "af_sarah" },
  { label: "American Female Sky", value: "af_sky" },
  { label: "American Male Adam", value: "am_adam" },
  { label: "American Male Echo", value: "am_echo" },
  { label: "American Male Eric", value: "am_eric" },
  { label: "American Male Fenrir", value: "am_fenrir" },
  { label: "American Male Liam", value: "am_liam" },
  { label: "American Male Michael", value: "am_michael" },
  { label: "American Male Onyx", value: "am_onyx" },
  { label: "American Male Puck", value: "am_puck" },
  { label: "British Female Alice", value: "bf_alice" },
  { label: "British Female Emma", value: "bf_emma" },
  { label: "British Female Isabella", value: "bf_isabella" },
  { label: "British Female Lily", value: "bf_lily" },
  { label: "British Male Daniel", value: "bm_daniel" },
  { label: "British Male Fable", value: "bm_fable" },
  { label: "British Male George", value: "bm_george" },
  { label: "British Male Lewis", value: "bm_lewis" },
];

/* -------------------- New: custom agent type -------------------- */
type CustomAgent = {
  inputFile?: File | null;
  localUrl?: string | null;
  serverPath?: string | null;
  voice: string;
  name: string;
  desc: string;
};

type StatusResp = {
  loaded: boolean;
  room?: string | null;
};

export interface PlaygroundMeta {
  name: string;
  value: string;
}

export interface PlaygroundProps {
  logo?: ReactNode;
  themeColors: string[];
  onConnect: (connect: boolean, opts?: { token: string; url: string }) => void;
}

const headerHeight = 56;

export default function Playground({
  logo,
  themeColors,
  onConnect,
}: PlaygroundProps) {
  const { config, setUserSettings } = useConfig();
  const { name } = useRoomInfo();
  const [transcripts, setTranscripts] = useState<ChatMessageType[]>([]);
  const { localParticipant } = useLocalParticipant();

  const voiceAssistant = useVoiceAssistant();

  const roomState = useConnectionState();
  const tracks = useTracks();
  const room = useRoomContext();

  const [rpcMethod, setRpcMethod] = useState("");
  const [rpcPayload, setRpcPayload] = useState("");
  const [showRpc, setShowRpc] = useState(false);
  const [transcribeText, setTranscribeText] = useState("");

  const agentOptions = [
    {
      name: "Alice",
      desc: "Interview Assistant",
      image: "alice.png",
      input_image: "static/avatar.png",
      voice: "af_heart",
    },
    {
      name: "Elenora",
      desc: "Customer Service",
      image: "elenora.png",
      input_image: "static/idle.mp4",
      voice: "af_sky",
    },
    {
      name: "James",
      desc: "Coding Assistant",
      image: "james.png",
      input_image: "static/james.mp4",
      voice: "am_adam",
    },
  ];
  const [selectedAgent, setSelectedAgent] = useState<(typeof agentOptions)[number] | null>(agentOptions[0]);

  /* -------------------- New: create-your-own modal state -------------------- */
  const [showCreateModal, setShowCreateModal] = useState(false);
  const [pendingCustom, setPendingCustom] = useState<CustomAgent | null>(null);

  const [serverLoaded, setServerLoaded] = useState(false);

  const checkLoaded = useCallback(async () => {
    try {
      const r = await fetch(API_URL + "/status");
      if (!r.ok) return;
      const s: StatusResp = await r.json();
      if (s.loaded) {
        setServerLoaded(true);
        setHasHeardAudio(true);
      }
    } catch {}
  }, []);

  /* -------------------- New: upload helper -------------------- */
  const tryUploadToServer = useCallback(async (file: File): Promise<string | null> => {
    try {
      const fd = new FormData();
      fd.append("file", file);
      const resp = await fetch(API_URL + "/upload", { method: "POST", body: fd });
      if (!resp.ok) throw new Error(`Upload http ${resp.status}`);
      const j = await resp.json();
      // backend returns {server_path, public_url}
      return (j?.public_url as string) || (j?.server_path ? `/${j.server_path}` : null);
    } catch (e) {
      console.warn("Upload failed or /upload not available. Falling back to local object URL.", e);
      return null;
    }
  }, []);

  /* -------------------- Offer / Stop with custom override -------------------- */
  useEffect(() => {
    if (roomState === ConnectionState.Connected && name) {
      const sendOffer = async () => {
        let input_image = selectedAgent?.input_image ?? "static/avatar.png";
        let voice = selectedAgent?.voice ?? "af_heart";
        let cname = selectedAgent?.name ?? "Assistant";
        let cdesc = selectedAgent?.desc ?? "Realtime assistant";

        if (pendingCustom) {
          if (pendingCustom.inputFile) {
            const serverPath = await tryUploadToServer(pendingCustom.inputFile);
            if (serverPath) {
              pendingCustom.serverPath = serverPath;
            } else if (!pendingCustom.localUrl) {
              pendingCustom.localUrl = URL.createObjectURL(pendingCustom.inputFile);
            }
          }
          input_image = pendingCustom.serverPath || pendingCustom.localUrl || input_image;
          voice = pendingCustom.voice || voice;
          cname = pendingCustom.name || "Your Agent";
          cdesc = pendingCustom.desc || cdesc;
        }

        await fetch(API_URL + "/offer", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            room: name,
            input_image,
            voice,
            name: cname,
            desc: cdesc,
          }),
        })
          .then((res) => {
            if (!res.ok) throw new Error(`HTTP ${res.status}`);
            return res.json();
          })
          .then(() => {
            checkLoaded();
          });
      };
      sendOffer();
    }

    if (roomState === ConnectionState.Disconnected && name) {
      fetch(API_URL + "/stop", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ room: name }),
      })
        .then((res) => {
          if (!res.ok) throw new Error(`HTTP ${res.status}`);
          return res.json();
        })
        .then(() => {
          checkLoaded();
        });
    }
  }, [roomState, name, selectedAgent, pendingCustom, checkLoaded, tryUploadToServer]);

  const sendTranscription = useCallback(
    async (text: string) => {
      const trimmed = text.trim();
      if (!trimmed) return;
      if (!(roomState === ConnectionState.Connected)) return;

      const voiceToUse =
        pendingCustom?.voice ||
        selectedAgent?.voice ||
        "af_heart";

      await fetch(API_URL + "/speak", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          room: config.settings.room_name || name,
          text: trimmed,
          voice: voiceToUse,
        }),
      });
    },
    [config.settings.room_name, name, roomState, pendingCustom, selectedAgent],
  );

  useEffect(() => {
    if (roomState === ConnectionState.Connected) {
      localParticipant.setCameraEnabled(config.settings.inputs.camera);
      localParticipant.setMicrophoneEnabled(config.settings.inputs.mic);
    }
  }, [config, localParticipant, roomState]);

  const agentVideoTrack = tracks.find(
    (trackRef) =>
      trackRef.publication.kind === Track.Kind.Video &&
      trackRef.participant.isAgent,
  );

  const localTracks = tracks.filter(
    ({ participant }) => participant instanceof LocalParticipant,
  );
  const localCameraTrack = localTracks.find(({ source }) => source === Track.Source.Camera);
  const localScreenTrack = localTracks.find(({ source }) => source === Track.Source.ScreenShare);
  const localMicTrack = localTracks.find(({ source }) => source === Track.Source.Microphone);

  const onDataReceived = useCallback((msg: any) => {
    if (msg.topic === "transcription") {
      const decoded = JSON.parse(new TextDecoder("utf-8").decode(msg.payload));
      let timestamp = new Date().getTime();
      if ("timestamp" in decoded && decoded.timestamp > 0) {
        timestamp = decoded.timestamp;
      }
      setTranscripts((prev) => [
        ...prev,
        { name: "You", message: decoded.text, timestamp, isSelf: true },
      ]);
    } else if (msg.topic === "agent_audio_ready") {
      setHasHeardAudio(true);
      setServerLoaded(true);
    }
  }, []);
  useDataChannel(onDataReceived);

  const [hasHeardAudio, setHasHeardAudio] = useState(false);
  const videoWrapRef = useRef<HTMLDivElement | null>(null);

  useEffect(() => {
    if (roomState !== ConnectionState.Connected) {
      setHasHeardAudio(false);
      checkLoaded();
    }
  }, [roomState, checkLoaded]);

  useEffect(() => {
    checkLoaded();
  }, [checkLoaded]);

  const videoTileContent = useMemo(() => {
    const videoFitClassName = `object-${config.video_fit || "contain"}`;

    const disconnectedContent = (
      <div className="flex items-center justify-center text-gray-700 text-center w-full h-full">
        No agent video track. Connect to get started.
      </div>
    );

    const loadingContent = (
      <div className="flex flex-col items-center justify-center gap-2 text-gray-700 text-center h-full w-full">
        <img src="/loading.gif" alt="Loading..." />
        Waiting for agent video track…
      </div>
    );

    const videoContent = (
      <div ref={videoWrapRef} className="absolute inset-0 w-full h-full">
        {agentVideoTrack && (
          <VideoTrack
            trackRef={agentVideoTrack}
            className={`absolute inset-0 ${videoFitClassName} object-position-center w-full h-full`}
          />
        )}
        {!(hasHeardAudio || serverLoaded) && (
          <div className="absolute inset-0 flex flex-col items-center justify-center gap-2 text-gray-700 text-center bg-black">
            <img src="/loading.gif" alt="Loading..." />
            Waiting for agent voice…
          </div>
        )}
      </div>
    );

    let content = null;
    if (roomState === ConnectionState.Disconnected) {
      content = disconnectedContent;
    } else if (agentVideoTrack) {
      content = videoContent;
    } else {
      content = loadingContent;
    }

    return (
      <div className="flex flex-col w-full grow text-gray-950 bg-black rounded-sm border border-gray-800 relative">
        {content}
      </div>
    );
  }, [agentVideoTrack, config, roomState, hasHeardAudio, serverLoaded]);

  useEffect(() => {
    document.body.style.setProperty(
      "--lk-theme-color",
      // @ts-ignore
      tailwindTheme.colors[config.settings.theme_color]["500"],
    );
    document.body.style.setProperty(
      "--lk-drop-shadow",
      `var(--lk-theme-color) 0px 0px 18px`,
    );
  }, [config.settings.theme_color]);

  const audioTileContent = useMemo(() => {
    const disconnectedContent = (
      <div className="flex flex-col items-center justify-center gap-2 text-gray-700 text-center w-full">
        No agent audio track. Connect to get started.
      </div>
    );

    const waitingContent = (
      <div className="flex flex-col items-center gap-2 text-gray-700 text-center w-full">
        <LoadingSVG />
        Waiting for agent audio track…
      </div>
    );

    const visualizerContent = (
      <div
        className={`flex items-center justify-center w-full h-48 [--lk-va-bar-width:30px] [--lk-va-bar-gap:20px] [--lk-fg:var(--lk-theme-color)]`}
      >
        <BarVisualizer
          state={voiceAssistant.state}
          trackRef={voiceAssistant.audioTrack}
          barCount={5}
          options={{ minHeight: 20 }}
        />
      </div>
    );

    if (roomState === ConnectionState.Disconnected) {
      return disconnectedContent;
    }

    if (!voiceAssistant.audioTrack) {
      return waitingContent;
    }

    return visualizerContent;
  }, [
    voiceAssistant.audioTrack,
    config.settings.theme_color,
    roomState,
    voiceAssistant.state,
  ]);

  // put near other refs/state
  const firstReadyBySound = useRef(false);
  // re-arm first-hear-sound ONLY when the room actually disconnects
  useEffect(() => {
    if (!room) return;

    const onRoomDisconnected = () => {
      firstReadyBySound.current = false;
      setHasHeardAudio(false);
      setServerLoaded(false);
    };

    room.on(RoomEvent.Disconnected, onRoomDisconnected);
    return () => {
      room.off(RoomEvent.Disconnected, onRoomDisconnected);
    };
  }, [room, setHasHeardAudio, setServerLoaded]);

  // FIRST LOAD: unlock UI only when we detect real audio energy
  useEffect(() => {
    if (firstReadyBySound.current) return;
    if (hasHeardAudio || serverLoaded) return;

    const lkTrack: any = voiceAssistant.audioTrack?.publication?.track;
    if (!lkTrack) return;

    let audioEl: HTMLAudioElement | null = null;
    let ctx: AudioContext | null = null;
    let src: MediaStreamAudioSourceNode | null = null;
    let analyser: AnalyserNode | null = null;
    let raf = 0;
    let stop = false;

    const cleanup = () => {
      stop = true;
      if (raf) cancelAnimationFrame(raf);
      try { analyser?.disconnect(); } catch {}
      try { src?.disconnect(); } catch {}
      try { ctx?.close(); } catch {}
      try { audioEl && lkTrack?.detach?.(audioEl); } catch {}
      audioEl = null; ctx = null; src = null; analyser = null;
    };

    try {
      audioEl = lkTrack.attach() as HTMLAudioElement;
      audioEl.muted = true;
      audioEl.play().catch(() => {});

      const stream = (audioEl as any).srcObject as MediaStream | null;
      if (!stream) return cleanup;

      ctx = new (window.AudioContext || (window as any).webkitAudioContext)();
      ctx.resume?.().catch(() => {});
      src = ctx.createMediaStreamSource(stream);
      analyser = ctx.createAnalyser();
      analyser.fftSize = 512;
      src.connect(analyser);

      const buf = new Float32Array(analyser.fftSize);
      const tick = () => {
        if (stop || !analyser) return;
        analyser.getFloatTimeDomainData(buf);
        let e = 0;
        for (let i = 0; i < buf.length; i++) e += buf[i] * buf[i];
        e /= buf.length;

        if (e > 1e-5) {
          firstReadyBySound.current = true;
          setHasHeardAudio(true);
          setServerLoaded(true);
          cleanup();
          return;
        }
        raf = requestAnimationFrame(tick);
      };
      raf = requestAnimationFrame(tick);
    } catch {
      cleanup();
    }

    return cleanup;
  }, [voiceAssistant.audioTrack, hasHeardAudio, serverLoaded, setHasHeardAudio, setServerLoaded]);

  const chatTileContent = useMemo(() => {
    if (voiceAssistant.agent) {
      return (
        <TranscriptionTile
          agentAudioTrack={voiceAssistant.audioTrack}
          accentColor={config.settings.theme_color}
        />
      );
    }
    return <></>;
  }, [
    config.settings.theme_color,
    voiceAssistant.audioTrack,
    voiceAssistant.agent,
  ]);

  const handleRpcCall = useCallback(async () => {
    if (!voiceAssistant.agent || !room) {
      throw new Error("No agent or room available");
    }

    const response = await room.localParticipant.performRpc({
      destinationIdentity: voiceAssistant.agent.identity,
      method: rpcMethod,
      payload: rpcPayload,
    });
    return response;
  }, [room, rpcMethod, rpcPayload, voiceAssistant.agent]);

  const agentAttributes = useParticipantAttributes({
    participant: voiceAssistant.agent,
  });

  /* -------------------- UI: Settings tile with create button + modal -------------------- */
  const settingsTileContent = useMemo(() => {
    return (
      <div className="flex flex-col h-full w-full items-start overflow-y-auto">
        <ConfigurationPanelItem title="Transcribe">
          <ChatMessageInput
            placeholder="Type text…"
            accentColor={config.settings.theme_color}
            height={56}
            onSend={sendTranscription}
          />
        </ConfigurationPanelItem>

        <ConfigurationPanelItem title="Room">
          <div className="flex flex-col gap-2">
            <EditableNameValueRow
              name="Room name"
              value={
                roomState === ConnectionState.Connected
                  ? name
                  : config.settings.room_name
              }
              valueColor={`${config.settings.theme_color}-500`}
              onValueChange={(value) => {
                const newSettings = { ...config.settings };
                newSettings.room_name = value;
                setUserSettings(newSettings);
              }}
              placeholder="Auto"
              editable={roomState !== ConnectionState.Connected}
            />
            <NameValueRow
              name="Status"
              value={
                roomState === ConnectionState.Connecting ? (
                  <LoadingSVG diameter={16} strokeWidth={2} />
                ) : (
                  roomState.charAt(0).toUpperCase() + roomState.slice(1)
                )
              }
              valueColor={
                roomState === ConnectionState.Connected
                  ? `${config.settings.theme_color}-500`
                  : "gray-500"
              }
            />
          </div>
        </ConfigurationPanelItem>

        <ConfigurationPanelItem title="Agent">
          <div className="flex flex-row flex-wrap gap-3 pt-2 justify-start w-full">
            {agentOptions.map((agent, i) => {
              const selected = selectedAgent?.name === agent.name && !pendingCustom;
              return (
                <div
                  key={i}
                  className={`avatar-option ${selected ? "selected" : ""}`}
                  onClick={() => {
                    setPendingCustom(null);
                    setSelectedAgent(agent);
                  }}
                >
                  <img src={agent.image} className="avatar-img" />
                  <strong>{agent.name}</strong>
                  <br />
                  <small>{agent.desc}</small>
                </div>
              );
            })}
          </div>

          <button
            className="mt-4 w-full rounded-lg border border-gray-700 bg-gray-800 hover:bg-gray-700 text-white py-3 text-sm font-medium"
            onClick={() => setShowCreateModal(true)}
          >
            Create your own agent
          </button>

          {pendingCustom && (
            <div className="mt-2 text-xs text-gray-300">
              Using custom agent: <b>{pendingCustom.name || "Your Agent"}</b>{" "}
              {pendingCustom.serverPath ? "(uploaded)" : pendingCustom.localUrl ? "(local)" : ""}
            </div>
          )}
        </ConfigurationPanelItem>

        {roomState === ConnectionState.Connected && voiceAssistant.agent && (
          <RpcPanel
            config={config}
            rpcMethod={rpcMethod}
            rpcPayload={rpcPayload}
            setRpcMethod={setRpcMethod}
            setRpcPayload={setRpcPayload}
            handleRpcCall={handleRpcCall}
          />
        )}
        {localCameraTrack && (
          <ConfigurationPanelItem title="Camera" source={Track.Source.Camera}>
            <div className="relative">
              <VideoTrack
                className="rounded-sm border border-gray-800 opacity-70 w-full"
                trackRef={localCameraTrack}
              />
            </div>
          </ConfigurationPanelItem>
        )}
        {localMicTrack && (
          <ConfigurationPanelItem
            title="Microphone"
            source={Track.Source.Microphone}
          >
            <AudioInputTile trackRef={localMicTrack} />
          </ConfigurationPanelItem>
        )}
        {config.show_qr && (
          <div className="w-full">
            <ConfigurationPanelItem title="QR Code">
              <QRCodeSVG value={typeof window !== "undefined" ? window.location.href : ""} width="128" />
            </ConfigurationPanelItem>
          </div>
        )}
      </div>
    );
  }, [
    config.description,
    config.settings,
    config.show_qr,
    localParticipant,
    name,
    roomState,
    localCameraTrack,
    localScreenTrack,
    localMicTrack,
    themeColors,
    setUserSettings,
    voiceAssistant.agent,
    rpcMethod,
    rpcPayload,
    handleRpcCall,
    sendTranscription,
    selectedAgent,
    pendingCustom,
  ]);

  /* -------------------- Modal Component -------------------- */
  const CreateAgentModal = () => {
    const [file, setFile] = useState<File | null>(null);
    const [previewUrl, setPreviewUrl] = useState<string | null>(null);
    const [role, setRole] = useState<string>("");
    const [voice, setVoice] = useState<string>(VOICE_OPTIONS[0]?.value || "af_alloy");
    const [nameInput, setNameInput] = useState<string>("Your Agent");

    // Webcam / selfie state
    const [camOpen, setCamOpen] = useState(false);
    const videoRef = useRef<HTMLVideoElement | null>(null);
    const streamRef = useRef<MediaStream | null>(null);

    const normalizeRole = (s: string) => {
      const words = s.trim().split(/\s+/).filter(Boolean);
      if (words.length <= 30) return s;
      return words.slice(0, 30).join(" ");
    };

    useEffect(() => {
      return () => {
        if (previewUrl) URL.revokeObjectURL(previewUrl);
        if (streamRef.current) {
          streamRef.current.getTracks().forEach((t) => t.stop());
          streamRef.current = null;
        }
      };
    }, [previewUrl]);

    const onFile = (f: File | null) => {
      setFile(f);
      if (previewUrl) {
        URL.revokeObjectURL(previewUrl);
        setPreviewUrl(null);
      }
      if (f) setPreviewUrl(URL.createObjectURL(f));
    };

    const openCamera = async () => {
      try {
        if (streamRef.current) {
          streamRef.current.getTracks().forEach((t) => t.stop());
        }
        const stream = await navigator.mediaDevices.getUserMedia({
          video: { width: { ideal: 960 }, height: { ideal: 960 }, facingMode: "user" },
          audio: false,
        });
        streamRef.current = stream;
        if (videoRef.current) {
          videoRef.current.srcObject = stream;
          await videoRef.current.play().catch(() => {});
        }
        setCamOpen(true);
      } catch (e) {
        console.error("getUserMedia failed:", e);
        alert("Unable to access the camera. Please allow permission and try again.");
      }
    };

    const closeCamera = () => {
      if (streamRef.current) {
        streamRef.current.getTracks().forEach((t) => t.stop());
        streamRef.current = null;
      }
      setCamOpen(false);
    };

    const captureSelfie = async () => {
      const v = videoRef.current;
      if (!v) return;
      const w = v.videoWidth || 720;
      const h = v.videoHeight || 720;
      const size = Math.min(w, h);
      const sx = (w - size) / 2;
      const sy = (h - size) / 2;

      const canvas = document.createElement("canvas");
      canvas.width = 900;
      canvas.height = 900;
      const ctx = canvas.getContext("2d");
      if (!ctx) return;

      ctx.save();
      ctx.translate(canvas.width, 0);
      ctx.scale(-1, 1);
      ctx.drawImage(v, sx, sy, size, size, 0, 0, canvas.width, canvas.height);
      ctx.restore();

      const blob: Blob | null = await new Promise((res) =>
        canvas.toBlob((b) => res(b), "image/png", 0.95),
      );
      if (!blob) return;

      const selfieFile = new File([blob], "selfie.png", { type: "image/png" });
      onFile(selfieFile);
      closeCamera();
    };

    const confirm = () => {
      const desc = normalizeRole(role);
      setPendingCustom({
        inputFile: file || undefined,
        localUrl: previewUrl || undefined,
        serverPath: null,
        voice,
        name: nameInput.trim() || "Your Agent",
        desc: desc || "Custom realtime agent",
      });
      setSelectedAgent(null);
      setShowCreateModal(false);
    };

    const cancel = () => {
      setShowCreateModal(false);
    };

    return (
      <div className="fixed inset-0 z-[9999] flex items-center justify-center bg-black/60 p-4">
        <div className="w-full max-w-3xl rounded-xl bg-gray-900 border border-gray-700 shadow-2xl">
          <div className="p-5 border-b border-gray-800 flex items-center justify-between">
            <h3 className="text-white font-semibold">Create your own agent</h3>
            <button
              className="text-gray-400 hover:text-gray-200"
              onClick={cancel}
              aria-label="Close"
            >
              ✕
            </button>
          </div>

          <div className="p-5 space-y-5">
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
              <div className="rounded-lg overflow-hidden border border-gray-800 bg-black/40 h-72 md:h-80 flex items-center justify-center">
                <img src="/elenora.png" className="w-full h-full object-cover" alt="Sample" />
              </div>
              <div className="rounded-lg border border-gray-800 p-4 text-sm text-gray-300 leading-relaxed">
                <div className="font-semibold text-white mb-2">Quick guide for best results</div>
                <ul className="list-disc ml-5 space-y-1">
                  <li>Use a clear, front-facing photo or half-body video.</li>
                  <li>Avoid showing teeth in the reference pose.</li>
                  <li>Neutral background is preferred.</li>
                  <li>PNG image or MP4 loop works (short, stable clip).</li>
                </ul>
              </div>
            </div>

            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
              <div className="space-y-4">
                <div className="space-y-3">
                  <label className="block text-sm text-gray-300">Upload .png/.jpg or .mp4</label>
                  <input
                    type="file"
                    accept=".png,.jpg,.jpeg,.mp4"
                    onChange={(e) => onFile(e.target.files?.[0] || null)}
                    className="block w-full text-sm text-gray-200 file:mr-4 file:py-2 file:px-4 file:rounded-md file:border-0 file:text-sm file:font-semibold file:bg-gray-700 file:text-white hover:file:bg-gray-600"
                  />
                </div>

                {/* Selfie capture */}
                <div className="space-y-2">
                  <label className="block text-sm text-gray-300">Or take a selfie</label>
                  {!camOpen ? (
                    <button
                      onClick={openCamera}
                      className="w-full rounded-md px-4 py-2 bg-gray-800 hover:bg-gray-700 text-white text-sm border border-gray-700"
                    >
                      Open Camera
                    </button>
                  ) : (
                    <div className="space-y-2">
                      <div className="relative rounded-md overflow-hidden border border-gray-800">
                        <video
                          ref={videoRef}
                          playsInline
                          muted
                          autoPlay
                          className="w-full h-64 object-cover transform -scale-x-100 bg-black"
                        />
                      </div>
                      <div className="flex gap-2">
                        <button
                          onClick={captureSelfie}
                          className="rounded-md px-4 py-2 bg-violet-600 hover:bg-violet-500 text-white text-sm"
                        >
                          Capture Selfie
                        </button>
                        <button
                          onClick={closeCamera}
                          className="rounded-md px-4 py-2 bg-gray-800 hover:bg-gray-700 text-white text-sm border border-gray-700"
                        >
                          Close
                        </button>
                      </div>
                      <p className="text-xs text-gray-400">
                        Tip: hold a neutral expression, front-facing, half-body if possible.
                      </p>
                    </div>
                  )}
                </div>

                {previewUrl && (
                  <div className="rounded-md overflow-hidden border border-gray-800">
                    {file?.type.includes("video") ? (
                      <video src={previewUrl} className="w-full" controls muted loop />
                    ) : (
                      <img src={previewUrl} className="w-full" />
                    )}
                  </div>
                )}
              </div>

              <div className="space-y-3">
                <label className="block text-sm text-gray-300">Agent name</label>
                <input
                  type="text"
                  value={nameInput}
                  onChange={(e) => setNameInput(e.target.value)}
                  className="w-full rounded-md border border-gray-700 bg-gray-800 text-white p-2 text-sm"
                  placeholder="Your Agent"
                />

                <label className="block text-sm text-gray-300 mt-2">Role (max 30 words)</label>
                <textarea
                  value={role}
                  onChange={(e) => setRole(e.target.value)}
                  onBlur={(e) => setRole(normalizeRole(e.target.value))}
                  className="w-full rounded-md border border-gray-700 bg-gray-800 text-white p-2 text-sm h-20"
                  placeholder="e.g., Friendly customer support for HRIS product"
                />

                <label className="block text-sm text-gray-300 mt-2">Voice</label>
                <select
                  value={voice}
                  onChange={(e) => setVoice(e.target.value)}
                  className="w-full rounded-md border border-gray-700 bg-gray-800 text-white p-2 text-sm"
                >
                  {VOICE_OPTIONS.map((v) => (
                    <option key={v.value} value={v.value}>
                      {v.label}
                    </option>
                  ))}
                </select>
              </div>
            </div>
          </div>

          <div className="p-5 border-t border-gray-800 flex items-center justify-end gap-3">
            <button
              onClick={cancel}
              className="rounded-md px-4 py-2 bg-gray-800 hover:bg-gray-700 text-white text-sm border border-gray-700"
            >
              Cancel
            </button>
            <button
              onClick={confirm}
              className="rounded-md px-4 py-2 bg-violet-600 hover:bg-violet-500 text-white text-sm"
            >
              Confirm
            </button>
          </div>
        </div>
      </div>
    );
  };


  /* -------------------- Layout / tabs -------------------- */
  let mobileTabs: PlaygroundTab[] = [];
  if (config.settings.outputs.video) {
    mobileTabs.push({
      title: "Video",
      content: (
        <PlaygroundTile
          className="w-full h-full grow"
          childrenClassName="justify-center"
        >
          {videoTileContent}
        </PlaygroundTile>
      ),
    });
  }

  if (config.settings.outputs.audio) {
    mobileTabs.push({
      title: "Audio",
      content: (
        <PlaygroundTile
          className="w-full h-full grow"
          childrenClassName="justify-center"
        >
          {audioTileContent}
        </PlaygroundTile>
      ),
    });
  }

  mobileTabs.push({
    title: "Settings",
    content: (
      <PlaygroundTile
        padding={false}
        backgroundColor="gray-950"
        className="h-full w-full basis-1/4 items-start overflow-y-auto flex"
        childrenClassName="h-full grow items-start"
      >
        {settingsTileContent}
      </PlaygroundTile>
    ),
  });

  return (
    <>
      <PlaygroundHeader
        title={config.title}
        logo={logo}
        githubLink={config.github_link}
        height={headerHeight}
        accentColor={config.settings.theme_color}
        connectionState={roomState}
        onConnectClicked={() =>
          onConnect(roomState === ConnectionState.Disconnected)
        }
      />

      {showCreateModal && <CreateAgentModal />}

      <div
        className={`flex gap-4 py-4 grow w-full selection:bg-${config.settings.theme_color}-900`}
        style={{ height: `calc(100% - ${headerHeight}px)` }}
      >
        <div className="flex flex-col grow basis-1/2 gap-4 h-full lg:hidden">
          <PlaygroundTabbedTile
            className="h-full"
            tabs={mobileTabs}
            initialTab={mobileTabs.length - 1}
          />
        </div>
        <div
          className={`flex-col grow basis-1/2 gap-4 h-full hidden lg:${
            !config.settings.outputs.audio && !config.settings.outputs.video
              ? "hidden"
              : "flex"
          }`}
        >
          {config.settings.outputs.video && (
            <PlaygroundTile
              title="Agent Video"
              className="w-full h-full grow"
              childrenClassName="justify-center"
            >
              {videoTileContent}
            </PlaygroundTile>
          )}
        </div>

        <PlaygroundTile
          padding={false}
          backgroundColor="gray-950"
          className="h-full w-full basis-1/4 items-start overflow-y-auto hidden max-w-[480px] lg:flex"
          childrenClassName="h-full grow items-start"
        >
          {settingsTileContent}
        </PlaygroundTile>
      </div>
    </>
  );
}
