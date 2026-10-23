import React from 'react'
import { Inbox, ArrowRight } from 'lucide-react'
import { Link } from 'react-router-dom'

export function EmptyState({
  icon: Icon = Inbox,
  title = 'No items found',
  description,
  actionLabel,
  actionHref,
  onAction,
  secondaryLabel,
  secondaryHref,
  onSecondaryAction,
  className = '',
}) {
  return (
    <div
      className={`border border-mist px-8 py-12 md:px-12 text-center rounded-control bg-surface/40 flex flex-col items-center justify-center ${className}`}
    >
      <div className="w-12 h-12 rounded-full bg-mist/50 border border-mist flex items-center justify-center text-muted mb-4 shadow-inner">
        <Icon size={24} className="text-teal/80" />
      </div>
      <h3 className="font-display text-base font-semibold text-ink tracking-tight">
        {title}
      </h3>
      {description && (
        <p className="mt-1.5 text-xs text-muted max-w-sm leading-relaxed">
          {description}
        </p>
      )}

      {(actionLabel || secondaryLabel) && (
        <div className="mt-5 flex flex-wrap items-center justify-center gap-3">
          {actionLabel && (
            actionHref ? (
              <Link
                to={actionHref}
                className="btn-primary inline-flex items-center gap-1.5 text-xs font-semibold py-2 px-3.5"
              >
                <span>{actionLabel}</span>
                <ArrowRight size={13} />
              </Link>
            ) : (
              <button
                type="button"
                onClick={onAction}
                className="btn-primary inline-flex items-center gap-1.5 text-xs font-semibold py-2 px-3.5"
              >
                <span>{actionLabel}</span>
                <ArrowRight size={13} />
              </button>
            )
          )}

          {secondaryLabel && (
            secondaryHref ? (
              <Link
                to={secondaryHref}
                className="btn-secondary inline-flex items-center gap-1.5 text-xs font-medium py-2 px-3"
              >
                <span>{secondaryLabel}</span>
              </Link>
            ) : (
              <button
                type="button"
                onClick={onSecondaryAction}
                className="btn-secondary inline-flex items-center gap-1.5 text-xs font-medium py-2 px-3"
              >
                <span>{secondaryLabel}</span>
              </button>
            )
          )}
        </div>
      )}
    </div>
  )
}

export default EmptyState
