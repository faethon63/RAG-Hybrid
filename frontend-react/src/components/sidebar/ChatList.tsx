import { useEffect } from 'react';
import { useChatStore } from '../../stores/chatStore';
import { useProjectStore } from '../../stores/projectStore';
import { ChatIcon, TrashIcon, LoaderIcon } from '../common/icons';
import { truncate } from '../../utils/parseThinking';
import clsx from 'clsx';

export function ChatList() {
  const chats = useChatStore((s) => s.chats);
  const chatsLoading = useChatStore((s) => s.chatsLoading);
  const currentChatId = useChatStore((s) => s.currentChatId);
  const loadChats = useChatStore((s) => s.loadChats);
  const loadChat = useChatStore((s) => s.loadChat);
  const deleteChat = useChatStore((s) => s.deleteChat);
  const currentProject = useProjectStore((s) => s.currentProject);

  useEffect(() => {
    loadChats(currentProject);
  }, [currentProject, loadChats]);

  const filteredChats = chats.filter((chat) =>
    currentProject ? chat.project === currentProject : !chat.project
  );

  const handleDelete = async (e: React.MouseEvent, chatId: string) => {
    e.stopPropagation();
    if (confirm('Delete this chat?')) {
      await deleteChat(chatId);
      await loadChats(currentProject);
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
            onClick={() => loadChat(chat.id)}
            className={clsx(
              'group flex items-center gap-2 px-2 py-2 rounded-lg cursor-pointer transition-colors',
              chat.id === currentChatId
                ? 'bg-[var(--color-surface)]'
                : 'hover:bg-[var(--color-surface-hover)]'
            )}
          >
            <ChatIcon className="w-4 h-4 text-[var(--color-text-secondary)] flex-shrink-0" />
            <span className="flex-1 text-sm truncate">
              {truncate(chat.name, 30)}
            </span>
            <button
              onClick={(e) => handleDelete(e, chat.id)}
              className="opacity-0 group-hover:opacity-100 p-1 hover:bg-[var(--color-border)] rounded transition-all"
            >
              <TrashIcon className="w-3 h-3 text-[var(--color-text-secondary)]" />
            </button>
          </div>
        ))
      )}
    </div>
  );
}
