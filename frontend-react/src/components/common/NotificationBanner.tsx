import { useState, useEffect } from 'react';
import { isPushSupported, isPushSubscribed, subscribeToPush } from '../../utils/pushNotifications';

export function NotificationBanner() {
  const [show, setShow] = useState(false);
  const [subscribing, setSubscribing] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    // Only show if: push supported, not already subscribed, not dismissed before
    const dismissed = localStorage.getItem('push-banner-dismissed');
    if (dismissed) return;
    if (!isPushSupported()) return;

    isPushSubscribed().then((subscribed) => {
      if (!subscribed) setShow(true);
    });
  }, []);

  if (!show) return null;

  const handleEnable = async () => {
    setSubscribing(true);
    setError(null);
    try {
      const success = await subscribeToPush();
      if (success) {
        setShow(false);
      } else {
        // Check what went wrong
        const perm = typeof Notification !== 'undefined' ? Notification.permission : 'unsupported';
        if (perm === 'denied') {
          setError('Notifications blocked. Enable in browser settings.');
        } else {
          setError('Failed to enable. Try again.');
        }
      }
    } catch (err) {
      console.error('Notification enable failed:', err);
      setError('Connection error. Try again.');
    }
    setSubscribing(false);
  };

  const handleDismiss = () => {
    localStorage.setItem('push-banner-dismissed', 'true');
    setShow(false);
  };

  return (
    <div className="fixed bottom-4 left-1/2 -translate-x-1/2 z-50 bg-[var(--color-surface)] border border-[var(--color-border)] rounded-xl shadow-lg px-5 py-3 flex flex-col gap-2 max-w-md">
      <div className="flex items-center gap-4">
        <div className="flex-1 text-sm">
          <p className="font-medium text-[var(--color-text)]">Enable notifications?</p>
          <p className="text-[var(--color-text-secondary)] text-xs mt-0.5">
            Get daily briefings and reminders on your phone.
          </p>
        </div>
        <button
          onClick={handleEnable}
          disabled={subscribing}
          className="px-3 py-1.5 bg-blue-600 text-white text-sm rounded-lg hover:bg-blue-500 disabled:opacity-50 whitespace-nowrap"
        >
          {subscribing ? 'Enabling...' : 'Enable'}
        </button>
        <button
          onClick={handleDismiss}
          className="text-[var(--color-text-secondary)] hover:text-[var(--color-text)] text-lg leading-none"
          title="Dismiss"
        >
          &times;
        </button>
      </div>
      {error && (
        <p className="text-xs text-red-400">{error}</p>
      )}
    </div>
  );
}
