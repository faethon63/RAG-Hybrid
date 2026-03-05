import { useState, useRef, useEffect, useCallback } from 'react';
import { useChatStore } from '../../stores/chatStore';
import { useSettingsStore, MODE_OPTIONS } from '../../stores/settingsStore';
import { useProjectStore } from '../../stores/projectStore';
import { api } from '../../api/client';
import { SendIcon, LoaderIcon, UploadIcon, CloseIcon, FileIcon, MicIcon, StopIcon, MenuIcon } from '../common/icons';
import clsx from 'clsx';

// Color mapping for each mode
const MODE_COLORS: Record<string, string> = {
  auto: '#3b82f6',      // Blue - smart/balanced
  private: '#22c55e',   // Green - privacy/security
  research: '#a855f7',  // Purple - deep knowledge
  deep_agent: '#f97316', // Orange - AI agent
};

interface AttachedFile {
  id: string;
  name: string;
  type: string;
  size: number;
  data: string; // base64 for images, text content for documents
  isImage: boolean;
}

export function ChatInput() {
  const [input, setInput] = useState('');
  const [attachments, setAttachments] = useState<AttachedFile[]>([]);
  const [isDragging, setIsDragging] = useState(false);
  const [isRecording, setIsRecording] = useState(false);
  const [recordingTime, setRecordingTime] = useState(0);
  const [isTranscribing, setIsTranscribing] = useState(false);
  const textareaRef = useRef<HTMLTextAreaElement>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);
  const mediaRecorderRef = useRef<MediaRecorder | null>(null);
  const audioChunksRef = useRef<Blob[]>([]);
  const recordingTimerRef = useRef<ReturnType<typeof setInterval> | null>(null);
  const streamRef = useRef<MediaStream | null>(null);
  const audioContextRef = useRef<AudioContext | null>(null);
  const silenceStartRef = useRef<number | null>(null);
  const silenceRafRef = useRef<number | null>(null);

  const isLoading = useChatStore((s) => s.isLoading);
  const sendQuery = useChatStore((s) => s.sendQuery);
  const setLastInputWasVoice = useChatStore((s) => s.setLastInputWasVoice);
  const voiceConversationMode = useChatStore((s) => s.voiceConversationMode);
  const setVoiceConversationMode = useChatStore((s) => s.setVoiceConversationMode);
  const shouldAutoRecord = useChatStore((s) => s.shouldAutoRecord);
  const setShouldAutoRecord = useChatStore((s) => s.setShouldAutoRecord);
  const lastAttachedFiles = useChatStore((s) => s.lastAttachedFiles);
  const mode = useSettingsStore((s) => s.mode);
  const setMode = useSettingsStore((s) => s.setMode);
  const model = useSettingsStore((s) => s.model);
  const health = useSettingsStore((s) => s.health);
  const sidebarOpen = useSettingsStore((s) => s.sidebarOpen);
  const setSidebarOpen = useSettingsStore((s) => s.setSidebarOpen);
  const currentProject = useProjectStore((s) => s.currentProject);

  const ollamaAvailable = health?.services?.ollama ?? false;
  const availableModes = MODE_OPTIONS.filter(
    opt => !opt.requiresOllama || ollamaAvailable
  );

  // Check if we have context files from previous messages
  const hasContextFiles = lastAttachedFiles.length > 0 && attachments.length === 0;

  // Auto-resize textarea
  useEffect(() => {
    const textarea = textareaRef.current;
    if (textarea) {
      textarea.style.height = 'auto';
      textarea.style.height = `${Math.min(textarea.scrollHeight, 200)}px`;
    }
  }, [input]);

  const processFile = useCallback(async (file: File): Promise<AttachedFile | null> => {
    const isImage = file.type.startsWith('image/');
    const isPdf = file.type === 'application/pdf';
    const isText = file.type.startsWith('text/') ||
                   file.name.endsWith('.md') ||
                   file.name.endsWith('.json') ||
                   file.name.endsWith('.txt');

    if (!isImage && !isPdf && !isText) {
      alert(`Unsupported file type: ${file.type || file.name}`);
      return null;
    }

    const id = `file_${Date.now()}_${Math.random().toString(36).slice(2)}`;

    if (isImage) {
      // Convert image to base64
      return new Promise((resolve) => {
        const reader = new FileReader();
        reader.onload = (e) => {
          resolve({
            id,
            name: file.name,
            type: file.type,
            size: file.size,
            data: e.target?.result as string,
            isImage: true,
          });
        };
        reader.readAsDataURL(file);
      });
    } else if (isPdf) {
      // For PDF, we'll send it to the backend to extract text
      return new Promise((resolve) => {
        const reader = new FileReader();
        reader.onload = (e) => {
          resolve({
            id,
            name: file.name,
            type: file.type,
            size: file.size,
            data: e.target?.result as string, // base64
            isImage: false,
          });
        };
        reader.readAsDataURL(file);
      });
    } else {
      // Text file - read as text
      return new Promise((resolve) => {
        const reader = new FileReader();
        reader.onload = (e) => {
          resolve({
            id,
            name: file.name,
            type: file.type,
            size: file.size,
            data: e.target?.result as string,
            isImage: false,
          });
        };
        reader.readAsText(file);
      });
    }
  }, []);

  const handleFiles = useCallback(async (files: FileList | File[]) => {
    const fileArray = Array.from(files);
    const processed = await Promise.all(fileArray.map(processFile));
    const valid = processed.filter((f): f is AttachedFile => f !== null);
    setAttachments((prev) => [...prev, ...valid]);
  }, [processFile]);

  const handleDrop = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(false);
    if (e.dataTransfer.files.length > 0) {
      handleFiles(e.dataTransfer.files);
    }
  }, [handleFiles]);

  const handlePaste = useCallback((e: React.ClipboardEvent) => {
    const items = e.clipboardData.items;
    const files: File[] = [];

    for (const item of items) {
      if (item.kind === 'file') {
        const file = item.getAsFile();
        if (file) files.push(file);
      }
    }

    if (files.length > 0) {
      e.preventDefault();
      handleFiles(files);
    }
  }, [handleFiles]);

  const removeAttachment = (id: string) => {
    setAttachments((prev) => prev.filter((f) => f.id !== id));
  };

  // Voice recording
  const stopRecordingCleanup = useCallback(() => {
    if (recordingTimerRef.current) {
      clearInterval(recordingTimerRef.current);
      recordingTimerRef.current = null;
    }
    if (silenceRafRef.current) {
      cancelAnimationFrame(silenceRafRef.current);
      silenceRafRef.current = null;
    }
    silenceStartRef.current = null;
    if (audioContextRef.current) {
      audioContextRef.current.close().catch(() => {});
      audioContextRef.current = null;
    }
    if (streamRef.current) {
      streamRef.current.getTracks().forEach(t => t.stop());
      streamRef.current = null;
    }
    setIsRecording(false);
    setRecordingTime(0);
  }, []);

  const startRecording = useCallback(async () => {
    try {
      // Clean up any lingering previous stream before starting fresh
      if (streamRef.current) {
        streamRef.current.getTracks().forEach(t => t.stop());
        streamRef.current = null;
      }
      // Enable echo cancellation to prevent mic picking up speaker TTS output
      const stream = await navigator.mediaDevices.getUserMedia({
        audio: { echoCancellation: true, noiseSuppression: true, autoGainControl: true }
      });
      streamRef.current = stream;
      audioChunksRef.current = [];

      const mediaRecorder = new MediaRecorder(stream, {
        mimeType: MediaRecorder.isTypeSupported('audio/webm;codecs=opus')
          ? 'audio/webm;codecs=opus'
          : 'audio/webm',
      });
      mediaRecorderRef.current = mediaRecorder;

      mediaRecorder.ondataavailable = (e) => {
        if (e.data.size > 0) audioChunksRef.current.push(e.data);
      };

      mediaRecorder.onstop = async () => {
        stopRecordingCleanup();
        const blob = new Blob(audioChunksRef.current, { type: 'audio/webm' });
        if (blob.size < 100) return; // Empty recording

        setIsTranscribing(true);
        try {
          const result = await api.transcribeAudio(blob);
          if (result.text) {
            const text = result.text.trim();
            // "Stop" command — cancel TTS and don't send
            const stopWords = ['stop', 'stop.', 'stop!', 'shut up', 'be quiet', 'silence'];
            if (stopWords.includes(text.toLowerCase())) {
              window.speechSynthesis?.cancel();
              // Turn off voice conversation mode
              useChatStore.getState().setVoiceConversationMode(false);
              return;
            }
            // Auto-send the transcribed text
            const isVoiceConvo = useChatStore.getState().voiceConversationMode;
            setLastInputWasVoice(true);
            await sendQuery(text, mode, model, currentProject, undefined, isVoiceConvo);
          }
        } catch (err) {
          console.error('Transcription failed:', err);
          const msg = err instanceof Error ? err.message : 'Transcription failed';
          useChatStore.getState().setError(`Voice: ${msg}`);
        } finally {
          setIsTranscribing(false);
        }
      };

      mediaRecorder.start(250); // Collect chunks every 250ms
      setIsRecording(true);

      // Timer for elapsed display
      setRecordingTime(0);
      recordingTimerRef.current = setInterval(() => {
        setRecordingTime(t => t + 1);
      }, 1000);

      // Silence detection (only in voice conversation mode)
      if (useChatStore.getState().voiceConversationMode) {
        try {
          const audioCtx = new AudioContext();
          audioContextRef.current = audioCtx;
          const source = audioCtx.createMediaStreamSource(stream);
          const analyser = audioCtx.createAnalyser();
          analyser.fftSize = 512;
          source.connect(analyser);

          const dataArray = new Uint8Array(analyser.fftSize);
          const SILENCE_THRESHOLD = 12; // RMS below this = silence
          const SILENCE_DURATION_MS = 2000; // 2 seconds of silence before auto-stop
          const MIN_RECORD_MS = 1000; // Don't auto-stop before 1 second of recording
          const recordStartTime = Date.now();
          silenceStartRef.current = null;

          const checkSilence = () => {
            if (!audioContextRef.current) return; // Cleaned up

            analyser.getByteTimeDomainData(dataArray);
            // Calculate RMS
            let sum = 0;
            for (let i = 0; i < dataArray.length; i++) {
              const val = (dataArray[i] - 128) / 128;
              sum += val * val;
            }
            const rms = Math.sqrt(sum / dataArray.length) * 255;

            if (rms < SILENCE_THRESHOLD) {
              // Silence detected
              if (silenceStartRef.current === null) {
                silenceStartRef.current = Date.now();
              } else {
                const silenceDuration = Date.now() - silenceStartRef.current;
                const recordDuration = Date.now() - recordStartTime;
                if (silenceDuration >= SILENCE_DURATION_MS && recordDuration >= MIN_RECORD_MS) {
                  // Auto-stop after sustained silence
                  if (mediaRecorderRef.current && mediaRecorderRef.current.state !== 'inactive') {
                    mediaRecorderRef.current.stop();
                  }
                  return; // Stop the loop
                }
              }
            } else {
              // Sound detected — reset silence timer
              silenceStartRef.current = null;
            }

            silenceRafRef.current = requestAnimationFrame(checkSilence);
          };

          silenceRafRef.current = requestAnimationFrame(checkSilence);
        } catch (err) {
          console.warn('Silence detection unavailable:', err);
          // Non-critical — recording works fine without it
        }
      }
    } catch (err) {
      console.error('Mic access failed:', err);
      useChatStore.getState().setError('Microphone access denied. Check browser permissions.');
    }
  }, [mode, model, currentProject, sendQuery, setLastInputWasVoice, stopRecordingCleanup]);

  const stopRecording = useCallback(() => {
    if (mediaRecorderRef.current && mediaRecorderRef.current.state !== 'inactive') {
      mediaRecorderRef.current.stop();
    }
  }, []);

  // Cleanup on unmount
  useEffect(() => {
    return () => {
      stopRecordingCleanup();
      if (mediaRecorderRef.current && mediaRecorderRef.current.state !== 'inactive') {
        mediaRecorderRef.current.stop();
      }
    };
  }, [stopRecordingCleanup]);

  // Auto-record when voice conversation mode signals it (TTS finished speaking)
  useEffect(() => {
    if (!shouldAutoRecord) return;

    // If voice mode was turned off, just clear the flag
    if (!useChatStore.getState().voiceConversationMode) {
      console.log('[Voice] Auto-record: voice mode off, clearing flag');
      setShouldAutoRecord(false);
      return;
    }

    // Wait for conditions to be ready — DON'T clear the flag yet
    // so the effect re-triggers when isLoading/isTranscribing change
    if (isRecording || isLoading || isTranscribing) {
      console.log('[Voice] Auto-record: waiting (recording=%s, loading=%s, transcribing=%s)', isRecording, isLoading, isTranscribing);
      return;
    }

    // Conditions met — clear flag and start recording
    console.log('[Voice] Auto-record: conditions met, starting recording');
    setShouldAutoRecord(false);
    window.speechSynthesis?.cancel();
    const timer = setTimeout(() => {
      if (useChatStore.getState().voiceConversationMode) {
        startRecording().catch((err) => {
          console.error('[Voice] Auto-record failed to start:', err);
        });
      }
    }, 800);
    return () => clearTimeout(timer);
  }, [shouldAutoRecord, isRecording, isLoading, isTranscribing, setShouldAutoRecord, startRecording]);

  const handleSubmit = async () => {
    const query = input.trim();
    if ((!query && attachments.length === 0) || isLoading) return;

    const files = attachments.length > 0 ? attachments : undefined;
    setInput('');
    setAttachments([]);
    await sendQuery(query || 'Analyze this file', mode, model, currentProject, files);
  };

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSubmit();
    }
  };

  const formatFileSize = (bytes: number) => {
    if (bytes < 1024) return `${bytes} B`;
    if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
    return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
  };

  return (
    <div className="border-t border-[var(--color-border)] bg-[var(--color-background)] p-4">
      <div className="max-w-3xl mx-auto">
        {/* Attachments preview */}
        {attachments.length > 0 && (
          <div className="mb-3 flex flex-wrap gap-2">
            {attachments.map((file) => (
              <div
                key={file.id}
                className="relative group flex items-center gap-2 px-3 py-2 bg-[var(--color-surface)] rounded-lg border border-[var(--color-border)]"
              >
                {file.isImage ? (
                  <img
                    src={file.data}
                    alt={file.name}
                    className="w-10 h-10 object-cover rounded"
                  />
                ) : (
                  <FileIcon className="w-5 h-5 text-[var(--color-text-secondary)]" />
                )}
                <div className="flex flex-col">
                  <span className="text-sm truncate max-w-[150px]">{file.name}</span>
                  <span className="text-xs text-[var(--color-text-secondary)]">
                    {formatFileSize(file.size)}
                  </span>
                </div>
                <button
                  onClick={() => removeAttachment(file.id)}
                  className="absolute -top-2 -right-2 p-1 bg-[var(--color-surface)] border border-[var(--color-border)] rounded-full opacity-0 group-hover:opacity-100 transition-opacity"
                >
                  <CloseIcon className="w-3 h-3" />
                </button>
              </div>
            ))}
          </div>
        )}

        {/* Input area */}
        <div
          className={clsx(
            'relative flex items-end gap-2 bg-[var(--color-surface)] rounded-2xl border transition-colors',
            isDragging
              ? 'border-[var(--color-primary)] border-dashed bg-[var(--color-primary)]/5'
              : 'border-[var(--color-border)] focus-within:border-[var(--color-primary)]'
          )}
          onDragOver={(e) => {
            e.preventDefault();
            setIsDragging(true);
          }}
          onDragLeave={() => setIsDragging(false)}
          onDrop={handleDrop}
        >
          {/* Sidebar toggle - visible on mobile/tablet */}
          <button
            onClick={() => setSidebarOpen(!sidebarOpen)}
            className="p-3 text-[var(--color-text-secondary)] hover:text-[var(--color-text)] transition-colors lg:hidden"
            title={sidebarOpen ? 'Close sidebar' : 'Open sidebar'}
          >
            <MenuIcon className="w-5 h-5" />
          </button>

          {/* File upload button */}
          <button
            onClick={() => fileInputRef.current?.click()}
            disabled={isLoading}
            className="p-3 text-[var(--color-text-secondary)] hover:text-[var(--color-text)] transition-colors"
            title="Attach file"
          >
            <UploadIcon className="w-5 h-5" />
          </button>
          <input
            ref={fileInputRef}
            type="file"
            multiple
            accept="image/*,.pdf,.txt,.md,.json"
            onChange={(e) => e.target.files && handleFiles(e.target.files)}
            className="hidden"
          />

          {/* Recording indicator overlay */}
          {isRecording && (
            <div className="absolute inset-0 flex items-center justify-center bg-[var(--color-surface)] rounded-2xl border-2 border-red-500 z-10">
              <div className="flex items-center gap-3">
                <span className="w-3 h-3 bg-red-500 rounded-full animate-pulse" />
                <span className="text-red-400 font-medium">
                  {Math.floor(recordingTime / 60)}:{(recordingTime % 60).toString().padStart(2, '0')}
                </span>
                <button
                  onClick={stopRecording}
                  className="p-3 bg-red-500 text-white rounded-full hover:bg-red-600 transition-colors"
                  style={{ minWidth: 48, minHeight: 48 }}
                >
                  <StopIcon className="w-5 h-5" />
                </button>
              </div>
            </div>
          )}

          {/* Transcribing indicator */}
          {isTranscribing && (
            <div className="absolute inset-0 flex items-center justify-center bg-[var(--color-surface)] rounded-2xl border border-[var(--color-primary)] z-10">
              <div className="flex items-center gap-2 text-[var(--color-primary)]">
                <LoaderIcon className="w-5 h-5 animate-spin" />
                <span>Transcribing...</span>
              </div>
            </div>
          )}

          <textarea
            ref={textareaRef}
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyDown={handleKeyDown}
            onPaste={handlePaste}
            placeholder={isDragging ? 'Drop files here...' : 'Ask anything...'}
            disabled={isLoading || isRecording || isTranscribing}
            rows={1}
            className={clsx(
              'flex-1 bg-transparent py-3 text-[var(--color-text)] placeholder-[var(--color-text-secondary)]',
              'resize-none outline-none min-h-[48px] max-h-[200px]',
              (isLoading || isRecording || isTranscribing) && 'opacity-50'
            )}
          />

          {/* Mic button (single recording) */}
          <button
            onClick={isRecording ? stopRecording : startRecording}
            disabled={isLoading || isTranscribing}
            className={clsx(
              'p-2 m-1 rounded-lg transition-all flex-shrink-0',
              isRecording
                ? 'bg-red-500 text-white'
                : 'text-[var(--color-text-secondary)] hover:text-[var(--color-text)] hover:bg-[var(--color-surface-hover)]'
            )}
            title={isRecording ? 'Stop recording' : 'Voice input'}
            style={{ minWidth: 44, minHeight: 44 }}
          >
            {isRecording ? (
              <StopIcon className="w-5 h-5" />
            ) : (
              <MicIcon className="w-5 h-5" />
            )}
          </button>

          {/* Voice conversation toggle */}
          <button
            onClick={() => {
              const next = !voiceConversationMode;
              setVoiceConversationMode(next);
              if (next && !isRecording && !isLoading && !isTranscribing) {
                // Stop any TTS first, then start recording after a delay
                window.speechSynthesis?.cancel();
                console.log('[Voice] Toggle ON — starting first recording');
                setTimeout(() => {
                  startRecording().catch((err) => {
                    console.error('[Voice] Toggle ON — recording failed:', err);
                  });
                }, 600);
              }
              if (!next) {
                console.log('[Voice] Toggle OFF');
                setShouldAutoRecord(false); // Clear any pending auto-record
                if (isRecording) stopRecording();
                window.speechSynthesis?.cancel();
              }
            }}
            disabled={isLoading || isTranscribing}
            className={clsx(
              'px-2 py-1 m-1 rounded-lg transition-all flex-shrink-0 text-xs font-medium',
              voiceConversationMode
                ? 'bg-green-500/20 text-green-400 border border-green-500/40'
                : 'text-[var(--color-text-secondary)] hover:text-[var(--color-text)] hover:bg-[var(--color-surface-hover)]'
            )}
            title={voiceConversationMode ? 'Stop voice conversation' : 'Start voice conversation (hands-free)'}
            style={{ minHeight: 44 }}
          >
            {voiceConversationMode ? 'Voice ON' : 'Voice'}
          </button>

          {/* Send button */}
          <button
            onClick={handleSubmit}
            disabled={(!input.trim() && attachments.length === 0) || isLoading || isRecording || isTranscribing}
            className={clsx(
              'p-2 m-2 rounded-lg transition-all',
              (input.trim() || attachments.length > 0) && !isLoading && !isRecording && !isTranscribing
                ? 'bg-[var(--color-primary)] text-white hover:bg-[var(--color-primary-hover)]'
                : 'bg-transparent text-[var(--color-text-secondary)] cursor-not-allowed'
            )}
          >
            {isLoading ? (
              <LoaderIcon className="w-5 h-5 animate-spin" />
            ) : (
              <SendIcon className="w-5 h-5" />
            )}
          </button>
        </div>

        {/* Status bar */}
        <div className="mt-2 flex items-center justify-between text-xs text-[var(--color-text-secondary)]">
          <div className="flex items-center gap-3">
            {/* Mode selector dots */}
            <div className="flex items-center gap-3">
              {availableModes.map((opt) => {
                const isActive = mode === opt.value;
                const color = MODE_COLORS[opt.value] || '#6b7280';
                return (
                  <button
                    key={opt.value}
                    onClick={() => setMode(opt.value)}
                    className="flex flex-col items-center gap-0.5"
                    title={opt.description}
                  >
                    <span
                      className="block rounded-full transition-all duration-200"
                      style={{
                        backgroundColor: color,
                        width: isActive ? '10px' : '6px',
                        height: isActive ? '10px' : '6px',
                        boxShadow: isActive ? `0 0 6px ${color}` : 'none',
                        opacity: isActive ? 1 : 0.4,
                      }}
                    />
                    <span
                      className="text-[9px] leading-none transition-opacity"
                      style={{ opacity: isActive ? 0.8 : 0.4 }}
                    >
                      {opt.label}
                    </span>
                  </button>
                );
              })}
            </div>
            {attachments.length > 0 && (
              <>
                <span>|</span>
                <span className="text-[var(--color-primary)]">
                  {attachments.some(f => f.isImage) ? 'Vision (llava)' : 'Files attached'}
                </span>
              </>
            )}
            {hasContextFiles && (
              <>
                <span>|</span>
                <span className="text-green-500" title="Previous image/files available for follow-up questions">
                  Context ({lastAttachedFiles.length})
                </span>
              </>
            )}
            {currentProject && (
              <>
                <span>|</span>
                <span className="text-[var(--color-primary)]">{currentProject}</span>
              </>
            )}
          </div>
          <span className="opacity-50">Enter to send</span>
        </div>
      </div>
    </div>
  );
}
