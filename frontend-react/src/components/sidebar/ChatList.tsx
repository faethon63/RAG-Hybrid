import { useEffect, useState, useCallback } from 'react';
import { useChatStore } from '../../stores/chatStore';
import { useProjectStore } from '../../stores/projectStore';
import { api } from '../../api/client';
import { ChatIcon, TrashIcon, LoaderIcon, EditIcon, CheckIcon, CloseIcon } from '../common/icons';
import clsx from 'clsx';

export function ChatList() {
  const chats = useChatStore((s) => s.chats);
  const chatsLoading = useChatStore((s) => s.chatsLoading);
  const currentChatId = useChatStore((s) => s.currentChatId);
  const loadChats = useChatStore((s) => s.loadChats);
  const loadChat = useChatStore((s) => s.loadChat);
  const deleteChat = useChatStore((s) => s.deleteChat);
  const currentProject = useProjectStore((s) => s.currentProject);

  const [editingId, setEditingId] = useState<string | null>(null);
  const [editName, setEditName] = useState('');
  const [selectionMode, setSelectionMode] = useState(false);
  const [selectedIds, setSelectedIds] = useState<Set<string>>(new Set());
  const [bulkDeleting, setBulkDeleting] = useState(false);

  useEffect(() => {
    loadChats(currentProject);
  }, [currentProject, loadChats]);

  // Show project chats if project selected, or ALL chats when no project
  const filteredChats = currentProject
    ? chats.filter((chat) => chat.project === currentProject)
    : chats;

  const handleDelete = async (e: React.MouseEvent, chatId: string) => {
    e.stopPropagation();
    if (confirm('Delete this chat?')) {
      await deleteChat(chatId);
      await loadChats(currentProject);
    }
  };

  const handleStartRename = (e: React.MouseEvent, chatId: string, currentName: string) => {
    e.stopPropagation();
    setEditingId(chatId);
    setEditName(currentName);
  };

  const handleCancelRename = (e: React.MouseEvent) => {
    e.stopPropagation();
    setEditingId(null);
    setEditName('');
  };

  const handleSaveRename = async (e: React.MouseEvent, chatId: string) => {
    e.stopPropagation();
    if (editName.trim()) {
      await api.renameChat(chatId, editName.trim());
      await loadChats(currentProject);
    }
    setEditingId(null);
    setEditName('');
  };

  const handleKeyDown = (e: React.KeyboardEvent, chatId: string) => {
    if (e.key === 'Enter') {
      e.preventDefault();
      handleSaveRename(e as unknown as React.MouseEvent, chatId);
    } else if (e.key === 'Escape') {
      setEditingId(null);
      setEditName('');
    }
  };

  const toggleSelection = useCallback((e: React.MouseEvent, chatId: string) => {
    e.stopPropagation();
    setSelectedIds((prev) => {
      const next = new Set(prev);
      if (next.has(chatId)) {
        next.delete(chatId);
      } else {
        next.add(chatId);
      }
      // Exit selection mode if nothing selected
      if (next.size === 0) setSelectionMode(false);
      return next;
    });
  }, []);

  const enterSelectionMode = useCallback((e: React.MouseEvent, chatId: string) => {
    e.stopPropagation();
    e.preventDefault();
    setSelectionMode(true);
    setSelectedIds(new Set([chatId]));
  }, []);

  const exitSelectionMode = useCallback(() => {
    setSelectionMode(false);
    setSelectedIds(new Set());
  }, []);

  const selectAll = useCallback(() => {
    setSelectedIds(new Set(filteredChats.slice(0, 20).map((c) => c.id)));
  }, [filteredChats]);

  const handleBulkDelete = useCallback(async () => {
    if (selectedIds.size === 0) return;
    if (!confirm(`Delete ${selectedIds.size} chat${selectedIds.size > 1 ? 's' : ''}?`)) return;
    setBulkDeleting(true);
    for (const id of selectedIds) {
      await deleteChat(id);
    }
    await loadChats(currentProject);
    setSelectedIds(new Set());
    setSelectionMode(false);
    setBulkDeleting(false);
  }, [selectedIds, deleteChat, loadChats, currentProject]);

  // Escape exits selection mode
  useEffect(() => {
    if (!selectionMode) return;
    const handler = (e: KeyboardEvent) => {
      if (e.key === 'Escape') exitSelectionMode();
    };
    window.addEventListener('keydown', handler);
    return () => window.removeEventListener('keydown', handler);
  }, [selectionMode, exitSelectionMode]);

  if (chatsLoading) {
    return (
      <div className="flex items-center justify-center py-4">
        <LoaderIcon className="w-4 h-4 animate-spin text-[var(--color-text-secondary)]" />
      </div>
    );
  }

  return (
    <div className="space-y-1">
      {/* Selection mode toolbar */}
      {selectionMode && (
        <div className="sticky top-0 z-10 flex items-center gap-1 px-2 py-1.5 mb-1 bg-[var(--color-surface)] rounded-lg border border-[var(--color-border)]">
          <span className="text-xs text-[var(--color-text-secondary)] flex-1">
            {selectedIds.size} selected
          </span>
          <button
            onClick={selectAll}
            className="text-xs px-2 py-0.5 text-[var(--color-primary)] hover:bg-[var(--color-surface-hover)] rounded transition-colors"
          >
            All
          </button>
          <button
            onClick={handleBulkDelete}
            disabled={bulkDeleting || selectedIds.size === 0}
            className="flex items-center gap-1 text-xs px-2 py-0.5 text-red-400 hover:bg-red-500/10 rounded transition-colors disabled:opacity-50"
          >
            {bulkDeleting ? (
              <LoaderIcon className="w-3 h-3 animate-spin" />
            ) : (
              <TrashIcon className="w-3 h-3" />
            )}
            Delete
          </button>
          <button
            onClick={exitSelectionMode}
            className="p-0.5 hover:bg-[var(--color-border)] rounded transition-colors"
            title="Cancel"
          >
            <CloseIcon className="w-3 h-3 text-[var(--color-text-secondary)]" />
          </button>
        </div>
      )}

      {filteredChats.length === 0 ? (
        <div className="text-sm text-[var(--color-text-secondary)] py-2 px-2">
          No chats yet
        </div>
      ) : (
        filteredChats.slice(0, 20).map((chat) => (
          <div
            key={chat.id}
            onClick={() => {
              if (selectionMode) {
                toggleSelection({ stopPropagation: () => {} } as React.MouseEvent, chat.id);
              } else if (editingId !== chat.id) {
                loadChat(chat.id);
              }
            }}
            onContextMenu={(e) => { e.preventDefault(); enterSelectionMode(e, chat.id); }}
            className={clsx(
              'group flex items-center gap-2 px-2 py-2 rounded-lg cursor-pointer transition-colors',
              selectionMode && selectedIds.has(chat.id) && 'bg-[var(--color-primary)]/10',
              chat.id === currentChatId && !selectionMode
                ? 'border-l-2 border-[var(--color-primary)] font-medium'
                : 'hover:bg-[var(--color-surface-hover)]'
            )}
            style={chat.id === currentChatId && !selectionMode ? { backgroundColor: 'rgba(16, 163, 127, 0.2)' } : undefined}
          >
            {selectionMode ? (
              <input
                type="checkbox"
                checked={selectedIds.has(chat.id)}
                onChange={(e) => toggleSelection(e as unknown as React.MouseEvent, chat.id)}
                onClick={(e) => e.stopPropagation()}
                className="w-4 h-4 flex-shrink-0 accent-[var(--color-primary)]"
              />
            ) : (
              <ChatIcon className="w-4 h-4 text-[var(--color-text-secondary)] flex-shrink-0" />
            )}
            {editingId === chat.id && !selectionMode ? (
              <>
                <input
                  type="text"
                  value={editName}
                  onChange={(e) => setEditName(e.target.value)}
                  onKeyDown={(e) => handleKeyDown(e, chat.id)}
                  onClick={(e) => e.stopPropagation()}
                  autoFocus
                  className="flex-1 text-sm bg-[var(--color-background)] border border-[var(--color-border)] rounded px-1 py-0.5 outline-none focus:border-[var(--color-primary)]"
                />
                <button
                  onClick={(e) => handleSaveRename(e, chat.id)}
                  className="p-1 hover:bg-[var(--color-border)] rounded"
                >
                  <CheckIcon className="w-3 h-3 text-green-500" />
                </button>
                <button
                  onClick={handleCancelRename}
                  className="p-1 hover:bg-[var(--color-border)] rounded"
                >
                  <CloseIcon className="w-3 h-3 text-[var(--color-text-secondary)]" />
                </button>
              </>
            ) : (
              <>
                <span className="flex-1 text-sm truncate" title={chat.name}>
                  {chat.name}
                </span>
                {!selectionMode && (
                  <>
                    <button
                      onClick={(e) => handleStartRename(e, chat.id, chat.name)}
                      className="opacity-0 group-hover:opacity-100 p-1 hover:bg-[var(--color-border)] rounded transition-all"
                      title="Rename"
                    >
                      <EditIcon className="w-3 h-3 text-[var(--color-text-secondary)]" />
                    </button>
                    <button
                      onClick={(e) => handleDelete(e, chat.id)}
                      className="opacity-0 group-hover:opacity-100 p-1 hover:bg-[var(--color-border)] rounded transition-all"
                      title="Delete"
                    >
                      <TrashIcon className="w-3 h-3 text-[var(--color-text-secondary)]" />
                    </button>
                  </>
                )}
              </>
            )}
          </div>
        ))
      )}
    </div>
  );
}
