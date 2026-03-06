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
  const resumeRef = useRef<ReturnType<typeof setInterval> | null>(null);

  const clearTimers = useCallback(() => {
    if (watchdogRef.current) {
      clearTimeout(watchdogRef.current);
      watchdogRef.current = null;
    }
    if (resumeRef.current) {
      clearInterval(resumeRef.current);
      resumeRef.current = null;
    }
  }, []);

  const stop = useCallback(() => {
    if (isAvailable) {
      clearTimers();
      window.speechSynthesis.cancel();
      setIsSpeaking(false);
      utteranceRef.current = null;
    }
  }, [isAvailable, clearTimers]);

  const onEndRef = useRef<(() => void) | null>(null);

  const speak = useCallback((text: string, onEnd?: () => void) => {
    if (!isAvailable) return;

    // Stop any current speech and clear previous watchdog
    clearTimers();
    // Only cancel if something is actually playing — avoids Chrome firing
    // spurious 'canceled' error events on mobile when nothing is queued
    if (window.speechSynthesis.speaking || window.speechSynthesis.pending) {
      window.speechSynthesis.cancel();
    }

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
      clearTimers();
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
        clearTimers();
        setIsSpeaking(false);
        utteranceRef.current = null;
        onEndRef.current = null;
        return;
      }
      handleEnd();
    };

    window.speechSynthesis.speak(utterance);

    // Chrome Android kills speechSynthesis after ~15 seconds of continuous speech.
    // The pause/resume trick keeps the engine alive.
    resumeRef.current = setInterval(() => {
      if (window.speechSynthesis.speaking) {
        window.speechSynthesis.pause();
        window.speechSynthesis.resume();
      }
    }, 10000);

    // Watchdog: Chrome sometimes doesn't fire onend for long utterances.
    // Estimate ~80ms/char but cap at 60s so recovery is fast if TTS dies.
    const maxDuration = Math.min(Math.max(cleaned.length * 80, 8000) + 3000, 60000);
    watchdogRef.current = setTimeout(() => {
      if (utteranceRef.current === utterance) {
        console.warn('[TTS] Watchdog fired — onend did not fire within expected time');
        window.speechSynthesis.cancel();
        handleEnd();
      }
    }, maxDuration);
  }, [isAvailable, clearTimers]);

  // Ensure voices are loaded (async on some browsers)
  useEffect(() => {
    if (!isAvailable) return;
    // Chrome loads voices asynchronously
    window.speechSynthesis.getVoices();
    const handler = () => window.speechSynthesis.getVoices();
    window.speechSynthesis.addEventListener('voiceschanged', handler);
    return () => {
      window.speechSynthesis.removeEventListener('voiceschanged', handler);
      clearTimers();
      window.speechSynthesis.cancel();
    };
  }, [isAvailable]);

  return { speak, stop, isSpeaking, isAvailable };
}
