import { useState, useRef, useEffect } from 'react';
import { useChatStore } from '../../stores/chatStore';
import { useSettingsStore } from '../../stores/settingsStore';
import { useProjectStore } from '../../stores/projectStore';
import { SendIcon, LoaderIcon } from '../common/icons';
import clsx from 'clsx';

export function ChatInput() {
  const [input, setInput] = useState('');
  const textareaRef = useRef<HTMLTextAreaElement>(null);

  const isLoading = useChatStore((s) => s.isLoading);
  const sendQuery = useChatStore((s) => s.sendQuery);
  const mode = useSettingsStore((s) => s.mode);
  const model = useSettingsStore((s) => s.model);
  const currentProject = useProjectStore((s) => s.currentProject);

  // Auto-resize textarea
  useEffect(() => {
    const textarea = textareaRef.current;
    if (textarea) {
      textarea.style.height = 'auto';
      textarea.style.height = `${Math.min(textarea.scrollHeight, 200)}px`;
    }
  }, [input]);

  const handleSubmit = async () => {
    const query = input.trim();
    if (!query || isLoading) return;

    setInput('');
    await sendQuery(query, mode, model, currentProject);
  };

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSubmit();
    }
  };

  return (
    <div className="border-t border-[var(--color-border)] bg-[var(--color-background)] p-4">
      <div className="max-w-3xl mx-auto">
        <div className="relative flex items-end gap-2 bg-[var(--color-surface)] rounded-2xl border border-[var(--color-border)] focus-within:border-[var(--color-primary)] transition-colors">
          <textarea
            ref={textareaRef}
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyDown={handleKeyDown}
            placeholder="Ask anything..."
            disabled={isLoading}
            rows={1}
            className={clsx(
              'flex-1 bg-transparent px-4 py-3 text-[var(--color-text)] placeholder-[var(--color-text-secondary)]',
              'resize-none outline-none min-h-[48px] max-h-[200px]',
              isLoading && 'opacity-50'
            )}
          />
          <button
            onClick={handleSubmit}
            disabled={!input.trim() || isLoading}
            className={clsx(
              'p-2 m-2 rounded-lg transition-all',
              input.trim() && !isLoading
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
            <span className="capitalize">{mode}</span>
            <span>|</span>
            <span>{model === 'auto' ? 'Auto' : model}</span>
            {currentProject && (
              <>
                <span>|</span>
                <span className="text-[var(--color-primary)]">{currentProject}</span>
              </>
            )}
          </div>
          <div>
            <span className="opacity-50">Enter to send, Shift+Enter for new line</span>
          </div>
        </div>
      </div>
    </div>
  );
}
