import { useEffect, useState, useCallback, useRef } from 'react';
import { useChatStore } from '../../stores/chatStore';
import { useProjectStore } from '../../stores/projectStore';
import { api } from '../../api/client';
import { ChatIcon, TrashIcon, LoaderIcon, EditIcon, CheckIcon, CloseIcon, MoveIcon, SearchIcon } from '../common/icons';
import clsx from 'clsx';

export function ChatList() {
  const chats = useChatStore((s) => s.chats);
  const chatsLoading = useChatStore((s) => s.chatsLoading);
  const currentChatId = useChatStore((s) => s.currentChatId);
  const loadChats = useChatStore((s) => s.loadChats);
  const loadChat = useChatStore((s) => s.loadChat);
  const deleteChat = useChatStore((s) => s.deleteChat);
  const searchQuery = useChatStore((s) => s.searchQuery);
  const searchResults = useChatStore((s) => s.searchResults);
  const searchLoading = useChatStore((s) => s.searchLoading);
  const searchChats = useChatStore((s) => s.searchChats);
  const clearSearch = useChatStore((s) => s.clearSearch);
  const setSearchQuery = useChatStore((s) => s.setSearchQuery);
  const currentProject = useProjectStore((s) => s.currentProject);

  const [editingId, setEditingId] = useState<string | null>(null);
  const [editName, setEditName] = useState('');
  const [movingChatId, setMovingChatId] = useState<string | null>(null);
  const [selectionMode, setSelectionMode] = useState(false);
  const [selectedIds, setSelectedIds] = useState<Set<string>>(new Set());
  const [bulkDeleting, setBulkDeleting] = useState(false);
  const editRef = useRef<HTMLInputElement>(null);
  const searchDebounceRef = useRef<ReturnType<typeof setTimeout>>(undefined);
  const projects = useProjectStore((s) => s.projects);

  const isSearching = searchQuery.length >= 2;

  useEffect(() => {
    loadChats(currentProject);
  }, [currentProject, loadChats]);

  const handleSearchInput = useCallback((value: string) => {
    setSearchQuery(value);
    if (searchDebounceRef.current) clearTimeout(searchDebounceRef.current);
    if (value.length >= 2) {
      searchDebounceRef.current = setTimeout(() => searchChats(value), 300);
    }
  }, [searchChats, setSearchQuery]);

  // Click outside cancels edit mode
  useEffect(() => {
    if (!editingId) return;
    const handler = (e: MouseEvent) => {
      if (editRef.current && !editRef.current.parentElement?.contains(e.target as Node)) {
        setEditingId(null);
        setEditName('');
      }
    };
    document.addEventListener('mousedown', handler);
    return () => document.removeEventListener('mousedown', handler);
  }, [editingId]);

  // Click outside closes move dropdown
  useEffect(() => {
    if (!movingChatId) return;
    const handler = () => setMovingChatId(null);
    // Use setTimeout so the click that opened the dropdown doesn't immediately close it
    const id = setTimeout(() => document.addEventListener('click', handler), 0);
    return () => { clearTimeout(id); document.removeEventListener('click', handler); };
  }, [movingChatId]);

  const handleMoveChat = async (chatId: string, targetProject: string | null) => {
    try {
      await api.updateChat(chatId, { project: targetProject });
      await loadChats(currentProject);
    } catch (err) {
      console.error('Failed to move chat:', err);
    }
    setMovingChatId(null);
  };

  // Show project chats if project selected, or only unassigned chats otherwise
  const filteredChats = currentProject
    ? chats.filter((chat) => chat.project === currentProject)
    : chats.filter((chat) => !chat.project);

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

  const displayChats = isSearching ? searchResults : filteredChats;

  return (
    <div className="space-y-1">
      {/* Search input */}
      <div className="relative px-1 mb-1">
        <SearchIcon className="absolute left-3 top-1/2 -translate-y-1/2 w-3.5 h-3.5 text-[var(--color-text-secondary)] pointer-events-none" />
        <input
          type="text"
          value={searchQuery}
          onChange={(e) => handleSearchInput(e.target.value)}
          placeholder="Search chats..."
          className="w-full pl-8 pr-7 py-1.5 text-sm bg-[var(--color-background)] border border-[var(--color-border)] rounded-lg outline-none focus:border-[var(--color-primary)] placeholder:text-[var(--color-text-secondary)]"
        />
        {searchQuery && (
          <button
            onClick={() => { clearSearch(); }}
            className="absolute right-3 top-1/2 -translate-y-1/2 p-0.5 hover:bg-[var(--color-border)] rounded transition-colors"
          >
            <CloseIcon className="w-3 h-3 text-[var(--color-text-secondary)]" />
          </button>
        )}
        {searchLoading && (
          <LoaderIcon className="absolute right-3 top-1/2 -translate-y-1/2 w-3.5 h-3.5 animate-spin text-[var(--color-text-secondary)]" />
        )}
      </div>

      {/* Selection mode toolbar */}
      {!isSearching && selectionMode ? (
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
      ) : !isSearching && filteredChats.length > 0 && (
        <div className="flex justify-end mb-1">
          <button
            onClick={() => setSelectionMode(true)}
            className="text-xs text-[var(--color-text-secondary)] hover:text-[var(--color-text)] transition-colors"
          >
            Select
          </button>
        </div>
      )}

      {displayChats.length === 0 ? (
        <div className="text-sm text-[var(--color-text-secondary)] py-2 px-2">
          {isSearching ? 'No matching chats' : 'No chats yet'}
        </div>
      ) : (
        displayChats.slice(0, 20).map((chat) => (
          <div
            key={chat.id}
            onClick={() => {
              if (isSearching) {
                loadChat(chat.id);
                clearSearch();
              } else if (selectionMode) {
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
                  ref={editRef}
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
                <span className="flex-1 min-w-0">
                  <span className="text-sm truncate block" title={chat.name}>
                    {chat.name}
                  </span>
                  {isSearching && chat.project && (
                    <span className="text-[10px] text-[var(--color-text-secondary)] truncate block opacity-70">
                      {chat.project}
                    </span>
                  )}
                </span>
                {!selectionMode && !isSearching && (
                  <div className="flex items-center gap-0.5 opacity-0 group-hover:opacity-100 transition-opacity relative">
                    <button
                      onClick={(e) => handleStartRename(e, chat.id, chat.name)}
                      className="p-1 hover:bg-[var(--color-border)] rounded transition-colors"
                      title="Rename"
                    >
                      <EditIcon className="w-3 h-3 text-[var(--color-text-secondary)]" />
                    </button>
                    <button
                      onClick={(e) => { e.stopPropagation(); setMovingChatId(movingChatId === chat.id ? null : chat.id); }}
                      className="p-1 hover:bg-[var(--color-border)] rounded transition-colors"
                      title="Move to project"
                    >
                      <MoveIcon className="w-3 h-3 text-[var(--color-text-secondary)]" />
                    </button>
                    <button
                      onClick={(e) => handleDelete(e, chat.id)}
                      className="p-1 hover:bg-[var(--color-border)] rounded transition-colors"
                      title="Delete"
                    >
                      <TrashIcon className="w-3 h-3 text-[var(--color-text-secondary)]" />
                    </button>
                    {/* Move-to dropdown */}
                    {movingChatId === chat.id && (
                      <div
                        className="absolute right-0 top-full mt-1 z-30 bg-[var(--color-surface)] border border-[var(--color-border)] rounded-lg shadow-xl overflow-hidden min-w-[140px]"
                        onClick={(e) => e.stopPropagation()}
                      >
                        <div className="px-2 py-1.5 text-xs text-[var(--color-text-secondary)] font-medium">
                          Move to...
                        </div>
                        <button
                          onClick={() => handleMoveChat(chat.id, null)}
                          className={clsx(
                            'w-full text-left px-3 py-1.5 text-sm hover:bg-[var(--color-surface-hover)] transition-colors',
                            !chat.project && 'text-[var(--color-primary)]'
                          )}
                        >
                          No project
                        </button>
                        {projects.map((p) => (
                          <button
                            key={p.name}
                            onClick={() => handleMoveChat(chat.id, p.name)}
                            className={clsx(
                              'w-full text-left px-3 py-1.5 text-sm hover:bg-[var(--color-surface-hover)] transition-colors truncate',
                              chat.project === p.name && 'text-[var(--color-primary)]'
                            )}
                          >
                            {p.name}
                          </button>
                        ))}
                      </div>
                    )}
                  </div>
                )}
              </>
            )}
          </div>
        ))
      )}
    </div>
  );
}
