import { useState, useEffect, useCallback, useRef } from 'react';
import { useProjectStore } from '../../stores/projectStore';
import { CloseIcon, SaveIcon, LoaderIcon } from '../common/icons';

export function FileEditor() {
  const editingFile = useProjectStore((s) => s.editingFile);
  const editingFileLoading = useProjectStore((s) => s.editingFileLoading);
  const savingFile = useProjectStore((s) => s.savingFile);
  const closeFileEditor = useProjectStore((s) => s.closeFileEditor);
  const saveFileContent = useProjectStore((s) => s.saveFileContent);

  const [content, setContent] = useState('');
  const [modified, setModified] = useState(false);
  const textareaRef = useRef<HTMLTextAreaElement>(null);

  // Sync content when file loads
  useEffect(() => {
    if (editingFile) {
      setContent(editingFile.content);
      setModified(false);
    }
  }, [editingFile]);

  // Focus textarea on open
  useEffect(() => {
    if (editingFile && textareaRef.current) {
      textareaRef.current.focus();
    }
  }, [editingFile]);

  const handleSave = useCallback(async () => {
    if (!editingFile || savingFile) return;
    try {
      await saveFileContent(editingFile.project, editingFile.filename, content);
      setModified(false);
    } catch {
      // Error logged in store
    }
  }, [editingFile, content, savingFile, saveFileContent]);

  const handleClose = useCallback(() => {
    if (modified) {
      if (!window.confirm('You have unsaved changes. Discard them?')) return;
    }
    closeFileEditor();
  }, [modified, closeFileEditor]);

  // Keyboard shortcuts
  useEffect(() => {
    if (!editingFile) return;
    const handler = (e: KeyboardEvent) => {
      if ((e.ctrlKey || e.metaKey) && e.key === 's') {
        e.preventDefault();
        handleSave();
      }
      if (e.key === 'Escape') {
        handleClose();
      }
    };
    window.addEventListener('keydown', handler);
    return () => window.removeEventListener('keydown', handler);
  }, [editingFile, handleSave, handleClose]);

  if (!editingFile && !editingFileLoading) return null;

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/60">
      <div className="bg-[var(--color-surface)] border border-[var(--color-border)] rounded-xl shadow-2xl flex flex-col w-[90vw] max-w-4xl h-[85vh]">
        {/* Header */}
        <div className="flex items-center justify-between px-4 py-3 border-b border-[var(--color-border)]">
          <div className="flex items-center gap-2 min-w-0">
            <h2 className="text-sm font-medium truncate">
              {editingFileLoading ? 'Loading...' : editingFile?.filename}
            </h2>
            {modified && (
              <span className="text-xs px-1.5 py-0.5 rounded bg-yellow-500/20 text-yellow-400 flex-shrink-0">
                modified
              </span>
            )}
          </div>
          <div className="flex items-center gap-2">
            <button
              onClick={handleSave}
              disabled={!modified || savingFile}
              className="flex items-center gap-1.5 px-3 py-1.5 text-sm rounded-lg bg-[var(--color-primary)] text-white hover:bg-[var(--color-primary)]/80 disabled:opacity-40 disabled:cursor-not-allowed transition-colors"
            >
              {savingFile ? (
                <LoaderIcon className="w-4 h-4 animate-spin" />
              ) : (
                <SaveIcon className="w-4 h-4" />
              )}
              Save & Re-index
            </button>
            <button
              onClick={handleClose}
              className="p-1.5 hover:bg-[var(--color-border)] rounded-lg transition-colors"
              title="Close (Esc)"
            >
              <CloseIcon className="w-5 h-5" />
            </button>
          </div>
        </div>

        {/* Body */}
        {editingFileLoading ? (
          <div className="flex-1 flex items-center justify-center text-[var(--color-text-secondary)]">
            <LoaderIcon className="w-6 h-6 animate-spin mr-2" />
            Loading file...
          </div>
        ) : (
          <textarea
            ref={textareaRef}
            value={content}
            onChange={(e) => {
              setContent(e.target.value);
              setModified(e.target.value !== editingFile?.content);
            }}
            className="flex-1 w-full p-4 bg-transparent text-sm font-mono resize-none outline-none text-[var(--color-text)]"
            spellCheck={false}
          />
        )}

        {/* Footer */}
        <div className="px-4 py-2 border-t border-[var(--color-border)] text-xs text-[var(--color-text-secondary)] flex justify-between">
          <span>Ctrl+S to save</span>
          <span>{content.length.toLocaleString()} chars</span>
        </div>
      </div>
    </div>
  );
}
