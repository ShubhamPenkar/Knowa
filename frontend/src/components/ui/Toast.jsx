import React, { createContext, useCallback, useContext, useEffect, useRef, useState } from 'react'
import { CheckCircle2, AlertTriangle, Info, XCircle, X } from 'lucide-react'

const ToastContext = createContext(null)

const ICONS = {
  success: CheckCircle2,
  error: XCircle,
  warning: AlertTriangle,
  info: Info,
}

const COLORS = {
  success: { bg: 'bg-teal-soft/90', border: 'border-teal/40', text: 'text-teal', icon: 'text-teal' },
  error:   { bg: 'bg-coral-soft/90', border: 'border-coral/40', text: 'text-ink', icon: 'text-coral' },
  warning: { bg: 'bg-[#3d2b0f]/90', border: 'border-[#e5a00d]/40', text: 'text-ink', icon: 'text-[#f5b838]' },
  info:    { bg: 'bg-surface/95', border: 'border-mist', text: 'text-ink', icon: 'text-teal' },
}

let toastId = 0

function ToastItem({ toast, onDismiss }) {
  const [exiting, setExiting] = useState(false)
  const timerRef = useRef(null)
  const Icon = ICONS[toast.type] || ICONS.info
  const colors = COLORS[toast.type] || COLORS.info

  const dismiss = useCallback(() => {
    setExiting(true)
    setTimeout(() => onDismiss(toast.id), 280)
  }, [toast.id, onDismiss])

  useEffect(() => {
    const duration = toast.duration || 4000
    timerRef.current = setTimeout(dismiss, duration)
    return () => clearTimeout(timerRef.current)
  }, [dismiss, toast.duration])

  return (
    <div
      role="status"
      aria-live="polite"
      className={`
        flex items-start gap-3 px-4 py-3 rounded-control border backdrop-blur-md
        shadow-lg shadow-black/20 max-w-sm w-full pointer-events-auto
        ${colors.bg} ${colors.border}
        ${exiting ? 'toast-exit' : 'toast-enter'}
      `}
    >
      <Icon className={`${colors.icon} shrink-0 mt-0.5`} size={18} strokeWidth={2} />
      <div className="flex-1 min-w-0">
        {toast.title && (
          <p className={`text-sm font-semibold ${colors.text} leading-snug`}>{toast.title}</p>
        )}
        <p className={`text-sm ${colors.text} leading-relaxed ${toast.title ? 'mt-0.5 opacity-80' : ''}`}>
          {toast.message}
        </p>
      </div>
      <button
        type="button"
        onClick={dismiss}
        className="shrink-0 text-muted hover:text-ink transition-colors mt-0.5"
        aria-label="Dismiss"
      >
        <X size={14} />
      </button>
    </div>
  )
}

export function ToastProvider({ children }) {
  const [toasts, setToasts] = useState([])

  const addToast = useCallback(({ type = 'info', title, message, duration }) => {
    const id = ++toastId
    setToasts((prev) => [...prev.slice(-4), { id, type, title, message, duration }])
    return id
  }, [])

  const removeToast = useCallback((id) => {
    setToasts((prev) => prev.filter((t) => t.id !== id))
  }, [])

  const toast = useCallback(
    (message, opts = {}) => addToast({ message, ...opts }),
    [addToast]
  )

  toast.success = (message, opts) => addToast({ type: 'success', message, ...opts })
  toast.error = (message, opts) => addToast({ type: 'error', message, ...opts })
  toast.warning = (message, opts) => addToast({ type: 'warning', message, ...opts })
  toast.info = (message, opts) => addToast({ type: 'info', message, ...opts })

  return (
    <ToastContext.Provider value={toast}>
      {children}
      {/* Toast container — fixed bottom-right */}
      <div
        className="fixed bottom-5 right-5 z-[9999] flex flex-col-reverse gap-2 pointer-events-none"
        aria-label="Notifications"
      >
        {toasts.map((t) => (
          <ToastItem key={t.id} toast={t} onDismiss={removeToast} />
        ))}
      </div>
    </ToastContext.Provider>
  )
}

export function useToast() {
  const ctx = useContext(ToastContext)
  if (!ctx) throw new Error('useToast must be used within <ToastProvider>')
  return ctx
}
