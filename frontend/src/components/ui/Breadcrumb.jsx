import React from 'react'
import { Link } from 'react-router-dom'
import { ChevronRight, Home } from 'lucide-react'

export function Breadcrumb({ items = [], className = '' }) {
  if (!items || items.length === 0) return null

  return (
    <nav aria-label="Breadcrumb" className={`flex items-center gap-1.5 text-xs text-muted mb-4 ${className}`}>
      <Link
        to="/"
        className="hover:text-ink transition-colors flex items-center gap-1 text-muted"
        title="Home"
      >
        <Home size={13} />
      </Link>
      {items.map((item, index) => {
        const isLast = index === items.length - 1
        return (
          <React.Fragment key={index}>
            <ChevronRight size={12} className="text-muted/60 shrink-0" />
            {isLast || !item.to ? (
              <span className="text-ink font-medium truncate max-w-[200px]" aria-current="page">
                {item.label}
              </span>
            ) : (
              <Link
                to={item.to}
                className="hover:text-ink transition-colors truncate max-w-[150px]"
              >
                {item.label}
              </Link>
            )}
          </React.Fragment>
        )
      })}
    </nav>
  )
}

export default Breadcrumb
