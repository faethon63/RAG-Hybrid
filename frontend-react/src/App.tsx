import { useEffect } from 'react';
import { Sidebar } from './components/sidebar/Sidebar';
import { ChatContainer } from './components/chat/ChatContainer';
import { SettingsPanel } from './components/settings/SettingsPanel';
import { ProjectForm } from './components/sidebar/ProjectForm';
import { FileEditor } from './components/sidebar/FileEditor';
import { NotificationBanner } from './components/common/NotificationBanner';
import { useSettingsStore } from './stores/settingsStore';
import { useChatStore } from './stores/chatStore';
import { MenuIcon } from './components/common/icons';

function App() {
  const sidebarOpen = useSettingsStore((s) => s.sidebarOpen);
  const setSidebarOpen = useSettingsStore((s) => s.setSidebarOpen);
  const loadSettings = useSettingsStore((s) => s.loadSettings);
  const checkHealth = useSettingsStore((s) => s.checkHealth);
  const loadChats = useChatStore((s) => s.loadChats);

  useEffect(() => {
    loadSettings();
    checkHealth();
    loadChats();
  }, [loadSettings, checkHealth, loadChats]);

  return (
    <div className="h-screen flex bg-[var(--color-background)]">
      <Sidebar />
      <main className="flex-1 flex flex-col min-w-0">
        {!sidebarOpen && (
          <div className="p-2 border-b border-[var(--color-border)]">
            <button
              onClick={() => setSidebarOpen(true)}
              className="p-2 hover:bg-[var(--color-surface)] rounded-lg transition-colors"
              style={{ minHeight: 44, minWidth: 44 }}
            >
              <MenuIcon className="w-6 h-6" />
            </button>
          </div>
        )}
        <ChatContainer />
        {/* Floating sidebar button - bottom left on mobile/tablet when sidebar is closed */}
        {!sidebarOpen && (
          <button
            onClick={() => setSidebarOpen(true)}
            className="fixed bottom-20 left-3 z-30 p-3 bg-[var(--color-surface)] border border-[var(--color-border)] rounded-full shadow-lg hover:bg-[var(--color-surface-hover)] transition-colors lg:hidden"
            style={{ minHeight: 48, minWidth: 48 }}
          >
            <MenuIcon className="w-6 h-6" />
          </button>
        )}
      </main>
      <SettingsPanel />
      <ProjectForm />
      <FileEditor />
      <NotificationBanner />
    </div>
  );
}

export default App;
