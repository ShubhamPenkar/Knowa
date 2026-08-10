/** Shared auth field validation (login / signup). */

export const EMAIL_RE = /^[^\s@]+@[^\s@]+\.[^\s@]+$/

export function validateEmail(value) {
  const v = String(value || '').trim()
  if (!v) return 'Enter your email'
  if (!EMAIL_RE.test(v)) return 'Enter a valid email address'
  return ''
}

export function validatePassword(value, { minLength = 1 } = {}) {
  const raw = String(value || '')
  if (!raw.trim()) return 'Enter your password'
  if (minLength > 1 && raw.length < minLength) {
    return `Password must be at least ${minLength} characters`
  }
  return ''
}

export function validateRequired(value, label = 'This field') {
  if (!String(value || '').trim()) return `Enter ${label}`
  return ''
}
