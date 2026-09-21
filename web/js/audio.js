window.JarvisAudio = (() => {
  let audioEnabled = true;
  let micEnabled = true;

  let audioContext = null;
  let mediaStream = null;
  let mediaSource = null;
  let analyser = null;
  let gainNode = null;
  let processor = null;
  let lowpassNode = null;
  let highpassNode = null;
  let masterOutputGain = null;

  let micLevel = 0;
  let micGain = 1.0;
  let levelCallback = null;
  let wsProvider = null;

  let noiseGate = 0.01;
  let speechThreshold = 0.02;

  const TARGET_SAMPLE_RATE = 16000;
  let playbackScheduledTime = 0;

  async function ensureAudioContext() {
    if (!audioContext) {
      audioContext = new (window.AudioContext || window.webkitAudioContext)();
    }
    if (!masterOutputGain) {
      masterOutputGain = audioContext.createGain();
      masterOutputGain.gain.value = audioEnabled ? 1.0 : 0.0;
      masterOutputGain.connect(audioContext.destination);
    }
    if (audioContext.state === "suspended") {
      try {
        await audioContext.resume();
      } catch (err) {
        console.warn("AudioContext resume failed:", err);
      }
    }
    return audioContext;
  }

  function setAudioEnabled(enabled) {
    audioEnabled = !!enabled;
    if (masterOutputGain && audioContext) {
      masterOutputGain.gain.cancelScheduledValues(audioContext.currentTime);
      masterOutputGain.gain.setValueAtTime(audioEnabled ? 1.0 : 0.0, audioContext.currentTime);
    }
    if (!audioEnabled) {
      playbackScheduledTime = 0;
    }
  }

  function setMicEnabled(enabled) {
    micEnabled = !!enabled;
  }

  function isAudioEnabled() {
    return audioEnabled;
  }

  function isMicEnabled() {
    return micEnabled;
  }

  function setMicGain(percent) {
    const clamped = Math.max(0, Math.min(300, Number(percent) || 100));
    micGain = clamped / 100;
    if (gainNode) {
      gainNode.gain.value = micGain;
    }
  }

  function setNoiseGatePercent(percent) {
    const value = Math.max(0, Math.min(8, Number(percent) || 0));
    noiseGate = value / 100;
  }

  function setSpeechThresholdPercent(percent) {
    const value = Math.max(0, Math.min(12, Number(percent) || 0));
    speechThreshold = value / 100;
  }

  function getMicGain() {
    return Math.round(micGain * 100);
  }

  function getMicLevel() {
    return micLevel;
  }

  function onMicLevel(callback) {
    levelCallback = callback;
  }

  function setSocketProvider(fn) {
    wsProvider = fn;
  }

  function downsampleBuffer(float32Data, inputSampleRate, outputSampleRate) {
    if (outputSampleRate >= inputSampleRate) {
      return float32Data;
    }

    const sampleRateRatio = inputSampleRate / outputSampleRate;
    const newLength = Math.round(float32Data.length / sampleRateRatio);
    const result = new Float32Array(newLength);

    let offsetResult = 0;
    let offsetBuffer = 0;

    while (offsetResult < result.length) {
      const nextOffsetBuffer = Math.round((offsetResult + 1) * sampleRateRatio);
      let accum = 0;
      let count = 0;

      for (let i = offsetBuffer; i < nextOffsetBuffer && i < float32Data.length; i++) {
        accum += float32Data[i];
        count++;
      }

      result[offsetResult] = count > 0 ? accum / count : 0;
      offsetResult++;
      offsetBuffer = nextOffsetBuffer;
    }

    return result;
  }

  function floatToPCM16(float32Data) {
    const pcm16 = new Int16Array(float32Data.length);
    for (let i = 0; i < float32Data.length; i++) {
      let sample = float32Data[i];
      if (sample > 1) sample = 1;
      if (sample < -1) sample = -1;
      pcm16[i] = sample < 0 ? sample * 0x8000 : sample * 0x7fff;
    }
    return pcm16;
  }

  function pcm16ToFloat32(pcmBytes) {
    const int16 = new Int16Array(pcmBytes);
    const float32 = new Float32Array(int16.length);
    for (let i = 0; i < int16.length; i++) {
      float32[i] = int16[i] / 32768;
    }
    return float32;
  }

  function scheduleFloat32Playback(float32Data, sampleRate) {
    if (!audioContext || !float32Data.length || !masterOutputGain) return;

    const buffer = audioContext.createBuffer(1, float32Data.length, sampleRate);
    buffer.copyToChannel(float32Data, 0, 0);

    const source = audioContext.createBufferSource();
    source.buffer = buffer;

    const outGain = audioContext.createGain();
    outGain.gain.value = 1.2;

    source.connect(outGain);
    outGain.connect(masterOutputGain);

    const now = audioContext.currentTime;
    if (playbackScheduledTime < now) {
      playbackScheduledTime = now + 0.03;
    }

    source.start(playbackScheduledTime);
    playbackScheduledTime += buffer.duration;
  }

  async function initMic() {
    if (audioContext && mediaStream) {
      await ensureAudioContext();
      return true;
    }

    await ensureAudioContext();

    mediaStream = await navigator.mediaDevices.getUserMedia({
      audio: {
        echoCancellation: true,
        noiseSuppression: true,
        autoGainControl: false,
        channelCount: 1
      }
    });

    mediaSource = audioContext.createMediaStreamSource(mediaStream);

    highpassNode = audioContext.createBiquadFilter();
    highpassNode.type = "highpass";
    highpassNode.frequency.value = 85;

    lowpassNode = audioContext.createBiquadFilter();
    lowpassNode.type = "lowpass";
    lowpassNode.frequency.value = 4200;

    gainNode = audioContext.createGain();
    gainNode.gain.value = micGain;

    analyser = audioContext.createAnalyser();
    analyser.fftSize = 2048;
    analyser.smoothingTimeConstant = 0.8;

    processor = audioContext.createScriptProcessor(4096, 1, 1);

    mediaSource.connect(highpassNode);
    highpassNode.connect(lowpassNode);
    lowpassNode.connect(gainNode);
    gainNode.connect(analyser);
    analyser.connect(processor);

    const silentGain = audioContext.createGain();
    silentGain.gain.value = 0.0;
    processor.connect(silentGain);
    silentGain.connect(audioContext.destination);

    const timeData = new Uint8Array(analyser.fftSize);

    processor.onaudioprocess = (event) => {
      analyser.getByteTimeDomainData(timeData);

      let sumSquares = 0;
      for (let i = 0; i < timeData.length; i++) {
        const normalized = (timeData[i] - 128) / 128;
        sumSquares += normalized * normalized;
      }

      const rms = Math.sqrt(sumSquares / timeData.length);
      const boosted = Math.min(1, rms * 5.8 * Math.max(0.65, micGain));
      micLevel = micEnabled ? boosted : 0;

      if (levelCallback) {
        levelCallback({
          level: micLevel,
          gain: getMicGain(),
        });
      }

      if (!micEnabled) return;

      const socket = wsProvider ? wsProvider() : null;
      if (!socket || socket.readyState !== WebSocket.OPEN) return;

      const inputBuffer = event.inputBuffer.getChannelData(0);

      let peak = 0;
      let voicedSamples = 0;
      for (let i = 0; i < inputBuffer.length; i++) {
        const abs = Math.abs(inputBuffer[i] * micGain);
        if (abs > peak) peak = abs;
        if (abs > speechThreshold) voicedSamples++;
      }

      const voicedRatio = voicedSamples / inputBuffer.length;

      if (peak < noiseGate && voicedRatio < 0.01) return;
      if (peak < speechThreshold) return;

      const amplified = new Float32Array(inputBuffer.length);
      for (let i = 0; i < inputBuffer.length; i++) {
        let sample = inputBuffer[i] * micGain;
        if (Math.abs(sample) < noiseGate) {
          sample = 0;
        }
        if (sample > 1) sample = 1;
        if (sample < -1) sample = -1;
        amplified[i] = sample;
      }

      const resampled = downsampleBuffer(amplified, audioContext.sampleRate, TARGET_SAMPLE_RATE);
      const pcm16 = floatToPCM16(resampled);

      socket.send(pcm16.buffer);
    };

    return true;
  }

  async function playPCM16(pcmBytes, sampleRate = 22050) {
    if (!pcmBytes || !pcmBytes.byteLength) return;

    await ensureAudioContext();

    const float32 = pcm16ToFloat32(pcmBytes);
    scheduleFloat32Playback(float32, sampleRate);
  }

  async function unlockFromUserGesture() {
    await ensureAudioContext();
  }

  return {
    setAudioEnabled,
    setMicEnabled,
    isAudioEnabled,
    isMicEnabled,
    setMicGain,
    setNoiseGatePercent,
    setSpeechThresholdPercent,
    getMicGain,
    getMicLevel,
    onMicLevel,
    setSocketProvider,
    initMic,
    playPCM16,
    unlockFromUserGesture,
  };
})();
