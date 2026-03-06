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
import { syncPushSubscription } from './utils/pushNotifications';

function App() {
  const sidebarOpen = useSettingsStore((s) => s.sidebarOpen);
  const setSidebarOpen = useSettingsStore((s) => s.setSidebarOpen);
  const loadSettings = useSettingsStore((s) => s.loadSettings);
  const checkHealth = useSettingsStore((s) => s.checkHealth);
  const loadChats = useChatStore((s) => s.loadChats);
  const loadChat = useChatStore((s) => s.loadChat);

  useEffect(() => {
    loadSettings();
    checkHealth();
    loadChats();
    syncPushSubscription(); // Re-sync browser subscription with backend

    // Handle ?chat= deep link from notifications
    const handleChatParam = () => {
      const params = new URLSearchParams(window.location.search);
      const chatParam = params.get('chat');
      if (chatParam) {
        loadChat(chatParam);
        window.history.replaceState({}, '', '/');
        return true;
      }
      return false;
    };

    if (!handleChatParam()) {
      // Restore last open chat on refresh
      const savedChatId = localStorage.getItem('rag-currentChatId');
      if (savedChatId) loadChat(savedChatId);
    }

    // Listen for LOAD_CHAT messages from service worker (notification deep link)
    const onSwMessage = (event: MessageEvent) => {
      if (event.data?.type === 'LOAD_CHAT' && event.data.chatId) {
        loadChat(event.data.chatId);
      }
    };
    navigator.serviceWorker?.addEventListener('message', onSwMessage);

    // Re-check URL when tab regains focus (fallback for openWindow case)
    const onVisibilityChange = () => {
      if (document.visibilityState === 'visible') handleChatParam();
    };
    document.addEventListener('visibilitychange', onVisibilityChange);
    return () => {
      navigator.serviceWorker?.removeEventListener('message', onSwMessage);
      document.removeEventListener('visibilitychange', onVisibilityChange);
    };
  }, [loadSettings, checkHealth, loadChats, loadChat]);

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
      </main>
      <SettingsPanel />
      <ProjectForm />
      <FileEditor />
      <NotificationBanner />
    </div>
  );
}

export default App;
