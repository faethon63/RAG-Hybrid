import { useSettingsStore } from '../../stores/settingsStore';
import { useChatStore } from '../../stores/chatStore';
import { useProjectStore } from '../../stores/projectStore';
import { ChatList } from './ChatList';
import { ProjectSelector } from './ProjectSelector';
import {
  PlusIcon,
  SettingsIcon,
  ChevronLeftIcon,
  RefreshIcon,
  CheckIcon,
  AlertIcon,
} from '../common/icons';
import clsx from 'clsx';

export function Sidebar() {
  const sidebarOpen = useSettingsStore((s) => s.sidebarOpen);
  const setSidebarOpen = useSettingsStore((s) => s.setSidebarOpen);
  const setShowSettings = useSettingsStore((s) => s.setShowSettings);
  const health = useSettingsStore((s) => s.health);
  const healthLoading = useSettingsStore((s) => s.healthLoading);
  const checkHealth = useSettingsStore((s) => s.checkHealth);
  const newChat = useChatStore((s) => s.newChat);
  const currentProject = useProjectStore((s) => s.currentProject);

  if (!sidebarOpen) {
    return null;
  }

  return (
    <aside className="w-64 bg-[var(--color-surface)] border-r border-[var(--color-border)] flex flex-col h-full">
      {/* Header */}
      <div className="p-3 border-b border-[var(--color-border)] flex items-center justify-between">
        <h1 className="font-semibold text-lg">RAG Hybrid</h1>
        <button
          onClick={() => setSidebarOpen(false)}
          className="p-1 hover:bg-[var(--color-surface-hover)] rounded transition-colors"
        >
          <ChevronLeftIcon className="w-5 h-5" />
        </button>
      </div>

      {/* New chat button */}
      <div className="p-3">
        <button
          onClick={newChat}
          className="w-full flex items-center justify-center gap-2 px-4 py-2 bg-[var(--color-primary)] hover:bg-[var(--color-primary-hover)] text-white rounded-lg transition-colors"
        >
          <PlusIcon className="w-4 h-4" />
          <span>New Chat</span>
        </button>
      </div>

      {/* Project selector */}
      <div className="px-3 pb-3">
        <ProjectSelector />
      </div>

      {/* Chat list */}
      <div className="flex-1 overflow-y-auto px-3">
        <div className="text-xs font-medium text-[var(--color-text-secondary)] uppercase tracking-wide mb-2">
          {currentProject ? `${currentProject} Chats` : 'Recent Chats'}
        </div>
        <ChatList />
      </div>

      {/* Footer */}
      <div className="p-3 border-t border-[var(--color-border)] space-y-2">
        {/* Health status */}
        <div className="flex items-center justify-between text-xs">
          <div className="flex items-center gap-2">
            {health?.status === 'healthy' ? (
              <CheckIcon className="w-3 h-3 text-green-500" />
            ) : health?.status === 'degraded' ? (
              <AlertIcon className="w-3 h-3 text-yellow-500" />
            ) : (
              <AlertIcon className="w-3 h-3 text-red-500" />
            )}
            <span className="text-[var(--color-text-secondary)] capitalize">
              {health?.status || 'Unknown'}
            </span>
          </div>
          <button
            onClick={checkHealth}
            disabled={healthLoading}
            className={clsx(
              'p-1 hover:bg-[var(--color-surface-hover)] rounded transition-colors',
              healthLoading && 'animate-spin'
            )}
          >
            <RefreshIcon className="w-3 h-3 text-[var(--color-text-secondary)]" />
          </button>
        </div>

        {/* Settings button */}
        <button
          onClick={() => setShowSettings(true)}
          className="w-full flex items-center gap-2 px-3 py-2 hover:bg-[var(--color-surface-hover)] rounded-lg transition-colors"
        >
          <SettingsIcon className="w-4 h-4 text-[var(--color-text-secondary)]" />
          <span className="text-sm">Settings</span>
        </button>
      </div>
    </aside>
  );
}
