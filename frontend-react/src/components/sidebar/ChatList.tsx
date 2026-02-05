import { useEffect, useState } from 'react';
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

  if (chatsLoading) {
    return (
      <div className="flex items-center justify-center py-4">
        <LoaderIcon className="w-4 h-4 animate-spin text-[var(--color-text-secondary)]" />
      </div>
    );
  }

  return (
    <div className="space-y-1">
      {filteredChats.length === 0 ? (
        <div className="text-sm text-[var(--color-text-secondary)] py-2 px-2">
          No chats yet
        </div>
      ) : (
        filteredChats.slice(0, 20).map((chat) => (
          <div
            key={chat.id}
            onClick={() => editingId !== chat.id && loadChat(chat.id)}
            className={clsx(
              'group flex items-center gap-2 px-2 py-2 rounded-lg cursor-pointer transition-colors',
              chat.id === currentChatId
                ? 'border-l-2 border-[var(--color-primary)] font-medium'
                : 'hover:bg-[var(--color-surface-hover)]'
            )}
            style={chat.id === currentChatId ? { backgroundColor: '#ff0000', color: '#ffffff' } : undefined}
          >
            <ChatIcon className="w-4 h-4 text-[var(--color-text-secondary)] flex-shrink-0" />
            {editingId === chat.id ? (
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
          </div>
        ))
      )}
    </div>
  );
}
