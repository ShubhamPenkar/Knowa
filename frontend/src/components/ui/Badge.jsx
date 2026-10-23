import React from 'react'
import { CheckCircle2, AlertTriangle, XCircle, Info, Clock, AlertCircle } from 'lucide-react'

const VARIANTS = {
  safe: {
    className: 'bg-teal-soft/80 border-teal/40 text-teal',
    icon: CheckCircle2,
  },
  caution: {
    className: 'bg-[#3d2b0f]/80 border-[#e5a00d]/40 text-[#f5b838]',
    icon: AlertTriangle,
  },
  risk: {
    className: 'bg-coral-soft/80 border-coral/40 text-coral',
    icon: XCircle,
  },
  urgent: {
    className: 'bg-coral-soft border-coral/50 text-coral font-bold',
    icon: AlertCircle,
  },
  info: {
    className: 'bg-mist/60 border-mist text-ink',
    icon: Info,
  },
  progress: {
    className: 'bg-[#3d2b0f]/80 border-[#e5a00d]/40 text-[#f5b838]',
    icon: Clock,
  },
  neutral: {
    className: 'bg-mist/40 border-mist/80 text-muted',
    icon: Info,
  },
}

export function Badge({
  variant = 'info',
  children,
  icon: customIcon,
  size = 'sm',
  className = '',
  showIcon = true,
}) {
  const conf = VARIANTS[variant] || VARIANTS.info
  const Icon = customIcon || conf.icon
  const iconSize = size === 'xs' ? 11 : 13

  const sizeClasses =
    size === 'xs'
      ? 'text-[11px] px-2 py-0.5 gap-1'
      : 'text-xs px-2.5 py-0.5 gap-1.5'

  return (
    <span
      className={`inline-flex items-center font-semibold rounded-full border tracking-wide ${conf.className} ${sizeClasses} ${className}`}
    >
      {showIcon && Icon && <Icon size={iconSize} className="shrink-0" />}
      <span>{children}</span>
    </span>
  )
}

export default Badge
