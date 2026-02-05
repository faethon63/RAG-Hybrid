import { useState, useRef, useEffect, useCallback } from 'react';
import { useChatStore } from '../../stores/chatStore';
import { useSettingsStore, MODE_OPTIONS } from '../../stores/settingsStore';
import { useProjectStore } from '../../stores/projectStore';
import { SendIcon, LoaderIcon, UploadIcon, CloseIcon, FileIcon } from '../common/icons';
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
  const textareaRef = useRef<HTMLTextAreaElement>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);

  const isLoading = useChatStore((s) => s.isLoading);
  const sendQuery = useChatStore((s) => s.sendQuery);
  const lastAttachedFiles = useChatStore((s) => s.lastAttachedFiles);
  const mode = useSettingsStore((s) => s.mode);
  const setMode = useSettingsStore((s) => s.setMode);
  const model = useSettingsStore((s) => s.model);
  const health = useSettingsStore((s) => s.health);
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

          <textarea
            ref={textareaRef}
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyDown={handleKeyDown}
            onPaste={handlePaste}
            placeholder={isDragging ? 'Drop files here...' : 'Ask anything... (paste images, drop files)'}
            disabled={isLoading}
            rows={1}
            className={clsx(
              'flex-1 bg-transparent py-3 text-[var(--color-text)] placeholder-[var(--color-text-secondary)]',
              'resize-none outline-none min-h-[48px] max-h-[200px]',
              isLoading && 'opacity-50'
            )}
          />
          <button
            onClick={handleSubmit}
            disabled={(!input.trim() && attachments.length === 0) || isLoading}
            className={clsx(
              'p-2 m-2 rounded-lg transition-all',
              (input.trim() || attachments.length > 0) && !isLoading
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
          <div>
            <span className="opacity-50">Enter to send</span>
          </div>
        </div>
      </div>
    </div>
  );
}
