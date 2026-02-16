import { useEffect } from 'react';
import { Sidebar } from './components/sidebar/Sidebar';
import { ChatContainer } from './components/chat/ChatContainer';
import { SettingsPanel } from './components/settings/SettingsPanel';
import { ProjectForm } from './components/sidebar/ProjectForm';
import { FileEditor } from './components/sidebar/FileEditor';
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
            >
              <MenuIcon className="w-5 h-5" />
            </button>
          </div>
        )}
        <ChatContainer />
      </main>
      <SettingsPanel />
      <ProjectForm />
      <FileEditor />
    </div>
  );
}

export default App;
