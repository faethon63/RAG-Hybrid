import { useState, useCallback, useEffect } from 'react';
import { useSettingsStore } from '../../stores/settingsStore';
import { useProjectStore } from '../../stores/projectStore';
import { useChatStore } from '../../stores/chatStore';
import { ChatList } from './ChatList';
import { ProjectSelector } from './ProjectSelector';
import {
  SettingsIcon,
  ChevronLeftIcon,
  RefreshIcon,
  CheckIcon,
  AlertIcon,
  PlusIcon,
} from '../common/icons';
import clsx from 'clsx';

export function Sidebar() {
  const sidebarOpen = useSettingsStore((s) => s.sidebarOpen);
  const setSidebarOpen = useSettingsStore((s) => s.setSidebarOpen);
  const sidebarWidth = useSettingsStore((s) => s.sidebarWidth);
  const setSidebarWidth = useSettingsStore((s) => s.setSidebarWidth);
  const setShowSettings = useSettingsStore((s) => s.setShowSettings);
  const health = useSettingsStore((s) => s.health);
  const healthLoading = useSettingsStore((s) => s.healthLoading);
  const checkHealth = useSettingsStore((s) => s.checkHealth);
  const currentProject = useProjectStore((s) => s.currentProject);
  const newChat = useChatStore((s) => s.newChat);

  const [isResizing, setIsResizing] = useState(false);

  const handleMouseDown = useCallback((e: React.MouseEvent) => {
    e.preventDefault();
    setIsResizing(true);
  }, []);

  useEffect(() => {
    if (!isResizing) return;

    const handleMouseMove = (e: MouseEvent) => {
      setSidebarWidth(e.clientX);
    };

    const handleMouseUp = () => {
      setIsResizing(false);
    };

    document.addEventListener('mousemove', handleMouseMove);
    document.addEventListener('mouseup', handleMouseUp);

    return () => {
      document.removeEventListener('mousemove', handleMouseMove);
      document.removeEventListener('mouseup', handleMouseUp);
    };
  }, [isResizing, setSidebarWidth]);

  if (!sidebarOpen) {
    return null;
  }

  return (
    <aside
      className="bg-[var(--color-surface)] border-r border-[var(--color-border)] flex flex-col h-full relative"
      style={{ width: sidebarWidth }}
    >
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

      {/* New Chat button */}
      <div className="p-3 border-b border-[var(--color-border)]">
        <button
          onClick={newChat}
          className="w-full flex items-center justify-center gap-2 px-3 py-2 bg-[var(--color-primary)] text-white rounded-lg hover:bg-[var(--color-primary-hover)] transition-colors"
        >
          <PlusIcon className="w-4 h-4" />
          <span className="text-sm font-medium">New Chat</span>
        </button>
      </div>

      {/* Project list */}
      <div className="px-3 py-2 border-b border-[var(--color-border)]">
        <ProjectSelector />
      </div>

      {/* Chat list */}
      <div className="flex-1 overflow-y-auto px-3 pt-2">
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

      {/* Resize handle */}
      <div
        onMouseDown={handleMouseDown}
        className={clsx(
          'absolute top-0 right-0 w-1 h-full cursor-ew-resize hover:bg-[var(--color-primary)] transition-colors',
          isResizing && 'bg-[var(--color-primary)]'
        )}
      />
    </aside>
  );
}
