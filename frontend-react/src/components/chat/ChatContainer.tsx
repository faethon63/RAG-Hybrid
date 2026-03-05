import { useRef, useEffect, useCallback } from 'react';
import { useChatStore } from '../../stores/chatStore';
import { useProjectStore } from '../../stores/projectStore';
import { useSettingsStore } from '../../stores/settingsStore';
import { MessageItem } from './MessageItem';
import { ChatInput } from './ChatInput';
import { LoaderIcon, SparklesIcon } from '../common/icons';

export function ChatContainer() {
  const messages = useChatStore((s) => s.messages);
  const isLoading = useChatStore((s) => s.isLoading);
  const error = useChatStore((s) => s.error);
  const editMessage = useChatStore((s) => s.editMessage);
  const editAndRegenerate = useChatStore((s) => s.editAndRegenerate);
  const deleteMessage = useChatStore((s) => s.deleteMessage);
  const lastInputWasVoice = useChatStore((s) => s.lastInputWasVoice);
  const setLastInputWasVoice = useChatStore((s) => s.setLastInputWasVoice);
  const voiceConversationMode = useChatStore((s) => s.voiceConversationMode);
  const currentProject = useProjectStore((s) => s.currentProject);
  const mode = useSettingsStore((s) => s.mode);
  const model = useSettingsStore((s) => s.model);
  const messagesEndRef = useRef<HTMLDivElement>(null);

  const handleEdit = useCallback((id: string, content: string, regenerate: boolean) => {
    if (regenerate) {
      editAndRegenerate(id, content, mode, model, currentProject);
    } else {
      editMessage(id, content, currentProject);
    }
  }, [editMessage, editAndRegenerate, mode, model, currentProject]);

  const handleDelete = useCallback((id: string) => {
    deleteMessage(id, currentProject);
  }, [deleteMessage, currentProject]);

  // Reset voice flag after assistant message arrives (so auto-play only fires once)
  // Skip reset when voiceConversationMode is on — it handles auto-play independently
  useEffect(() => {
    if (lastInputWasVoice && !voiceConversationMode && !isLoading && messages.length > 0 && messages[messages.length - 1].role === 'assistant') {
      const timer = setTimeout(() => setLastInputWasVoice(false), 500);
      return () => clearTimeout(timer);
    }
  }, [lastInputWasVoice, voiceConversationMode, isLoading, messages, setLastInputWasVoice]);

  // Scroll to bottom on new messages
  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages, isLoading]);

  return (
    <div className="flex flex-col h-full">
      {/* Messages area */}
      <div className="flex-1 overflow-y-auto">
        {messages.length === 0 ? (
          <div className="h-full flex flex-col items-center justify-center text-center p-8">
            <div className="w-16 h-16 rounded-full bg-[var(--color-surface)] flex items-center justify-center mb-4">
              <SparklesIcon className="w-8 h-8 text-[var(--color-primary)]" />
            </div>
            <h2 className="text-xl font-semibold mb-2">
              How can I help you today?
            </h2>
            <p className="text-[var(--color-text-secondary)] max-w-md text-sm sm:text-base">
              Search documents, browse the web, or reason through complex tasks.
            </p>
          </div>
        ) : (
          <div className="pb-4">
            {messages.map((message, index) => {
              // Auto-play TTS on the last assistant message when voice input was used
              const isLastAssistant = !isLoading
                && message.role === 'assistant'
                && index === messages.length - 1
                && (lastInputWasVoice || voiceConversationMode);
              return (
                <MessageItem
                  key={message.id}
                  message={message}
                  isFirstMessage={index === 0}
                  autoPlayTTS={isLastAssistant}
                  onEdit={handleEdit}
                  onDelete={handleDelete}
                />
              );
            })}

            {/* Loading indicator */}
            {isLoading && (
              <div className="py-6 px-4 md:px-8 bg-[var(--color-surface)]/30">
                <div className="max-w-3xl mx-auto flex gap-4">
                  <div className="w-8 h-8 rounded-full bg-[var(--color-surface)] border border-[var(--color-border)] flex items-center justify-center">
                    <LoaderIcon className="w-4 h-4 text-[var(--color-primary)] animate-spin" />
                  </div>
                  <div className="flex-1">
                    <div className="text-sm font-medium mb-1 text-[var(--color-text-secondary)]">
                      Assistant
                    </div>
                    <div className="flex items-center gap-2 text-[var(--color-text-secondary)]">
                      <div className="flex gap-1">
                        <span className="w-2 h-2 bg-[var(--color-primary)] rounded-full animate-bounce" style={{ animationDelay: '0ms' }} />
                        <span className="w-2 h-2 bg-[var(--color-primary)] rounded-full animate-bounce" style={{ animationDelay: '150ms' }} />
                        <span className="w-2 h-2 bg-[var(--color-primary)] rounded-full animate-bounce" style={{ animationDelay: '300ms' }} />
                      </div>
                      <span className="text-sm">Thinking...</span>
                    </div>
                  </div>
                </div>
              </div>
            )}

            {/* Error display */}
            {error && (
              <div className="py-4 px-4 md:px-8">
                <div className="max-w-3xl mx-auto">
                  <div className="p-4 bg-red-500/10 border border-red-500/20 rounded-lg text-red-400 text-sm">
                    {error}
                  </div>
                </div>
              </div>
            )}

            <div ref={messagesEndRef} />
          </div>
        )}
      </div>

      {/* Input area */}
      <ChatInput />
    </div>
  );
}
