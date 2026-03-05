import { useState, useEffect, useRef, useCallback } from 'react';

/** Strip markdown formatting for cleaner TTS output */
function stripMarkdown(text: string): string {
  return text
    // Remove code blocks
    .replace(/```[\s\S]*?```/g, '')
    // Remove inline code
    .replace(/`([^`]+)`/g, '$1')
    // Remove headers
    .replace(/^#{1,6}\s+/gm, '')
    // Remove bold/italic
    .replace(/\*{1,3}([^*]+)\*{1,3}/g, '$1')
    .replace(/_{1,3}([^_]+)_{1,3}/g, '$1')
    // Remove links, keep text
    .replace(/\[([^\]]+)\]\([^)]+\)/g, '$1')
    // Remove images
    .replace(/!\[([^\]]*)\]\([^)]+\)/g, '')
    // Remove HTML tags
    .replace(/<[^>]+>/g, '')
    // Remove horizontal rules
    .replace(/^[-*_]{3,}$/gm, '')
    // Remove blockquotes
    .replace(/^>\s*/gm, '')
    // Remove list markers
    .replace(/^[\s]*[-*+]\s+/gm, '')
    .replace(/^[\s]*\d+\.\s+/gm, '')
    // Clean up multiple newlines
    .replace(/\n{3,}/g, '\n\n')
    .trim();
}

export function useSpeechSynthesis() {
  const [isSpeaking, setIsSpeaking] = useState(false);
  const [isAvailable] = useState(() => typeof window !== 'undefined' && 'speechSynthesis' in window);
  const utteranceRef = useRef<SpeechSynthesisUtterance | null>(null);
  const watchdogRef = useRef<ReturnType<typeof setTimeout> | null>(null);

  const clearWatchdog = useCallback(() => {
    if (watchdogRef.current) {
      clearTimeout(watchdogRef.current);
      watchdogRef.current = null;
    }
  }, []);

  const stop = useCallback(() => {
    if (isAvailable) {
      clearWatchdog();
      window.speechSynthesis.cancel();
      setIsSpeaking(false);
      utteranceRef.current = null;
    }
  }, [isAvailable, clearWatchdog]);

  const onEndRef = useRef<(() => void) | null>(null);

  const speak = useCallback((text: string, onEnd?: () => void) => {
    if (!isAvailable) return;

    // Stop any current speech and clear previous watchdog
    clearWatchdog();
    window.speechSynthesis.cancel();

    const cleaned = stripMarkdown(text);
    if (!cleaned) return;

    onEndRef.current = onEnd || null;
    const utterance = new SpeechSynthesisUtterance(cleaned);
    utteranceRef.current = utterance;

    // Try to pick a good voice (Google/Samsung voices sound natural on Android)
    const voices = window.speechSynthesis.getVoices();
    const preferred = voices.find(v =>
      v.name.includes('Google') || v.name.includes('Samsung')
    ) || voices.find(v => v.lang.startsWith('en') && !v.localService) || voices[0];

    if (preferred) utterance.voice = preferred;
    utterance.rate = 1.0;
    utterance.pitch = 1.0;

    const handleEnd = () => {
      clearWatchdog();
      setIsSpeaking(false);
      utteranceRef.current = null;
      const cb = onEndRef.current;
      onEndRef.current = null;
      cb?.();
    };

    utterance.onstart = () => setIsSpeaking(true);
    utterance.onend = handleEnd;
    utterance.onerror = (e) => {
      // 'interrupted' is normal when cancel() is called, don't treat as error
      if (e.error === 'interrupted') {
        clearWatchdog();
        setIsSpeaking(false);
        utteranceRef.current = null;
        onEndRef.current = null;
        return;
      }
      handleEnd();
    };

    window.speechSynthesis.speak(utterance);

    // Watchdog: Chrome sometimes doesn't fire onend for long utterances.
    // Estimate max duration: ~80ms per char + 5s buffer, minimum 8s.
    const maxDuration = Math.max(cleaned.length * 80, 5000) + 3000;
    watchdogRef.current = setTimeout(() => {
      if (utteranceRef.current === utterance) {
        console.warn('[TTS] Watchdog fired — onend did not fire within expected time');
        window.speechSynthesis.cancel();
        handleEnd();
      }
    }, maxDuration);
  }, [isAvailable, clearWatchdog]);

  // Ensure voices are loaded (async on some browsers)
  useEffect(() => {
    if (!isAvailable) return;
    // Chrome loads voices asynchronously
    window.speechSynthesis.getVoices();
    const handler = () => window.speechSynthesis.getVoices();
    window.speechSynthesis.addEventListener('voiceschanged', handler);
    return () => {
      window.speechSynthesis.removeEventListener('voiceschanged', handler);
      clearWatchdog();
      window.speechSynthesis.cancel();
    };
  }, [isAvailable]);

  return { speak, stop, isSpeaking, isAvailable };
}
