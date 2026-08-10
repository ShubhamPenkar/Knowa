import { useState } from 'react'
import { useNavigate, Link } from 'react-router-dom'
import { useAuth } from '../context/AuthContext'
import { validateEmail, validatePassword } from '../lib/authValidation'

const LOOP = [
  { label: 'Predict', hint: 'Calibrated risk' },
  { label: 'Explain', hint: 'Why it moved' },
  { label: 'Act', hint: 'Ranked moves' },
  { label: 'Learn', hint: 'Outcomes feed back' },
]

export default function Login() {
  const [email, setEmail] = useState('')
  const [password, setPassword] = useState('')
  const [showPassword, setShowPassword] = useState(false)
  const [fieldErrors, setFieldErrors] = useState({ email: '', password: '' })
  const [touched, setTouched] = useState({ email: false, password: false })
  const [error, setError] = useState('')
  const [loading, setLoading] = useState(false)
  const { login } = useAuth()
  const navigate = useNavigate()

  const setEmailValue = (value) => {
    setEmail(value)
    if (error) setError('')
    if (touched.email) {
      setFieldErrors((prev) => ({ ...prev, email: validateEmail(value) }))
    }
  }

  const setPasswordValue = (value) => {
    setPassword(value)
    if (error) setError('')
    if (touched.password) {
      setFieldErrors((prev) => ({ ...prev, password: validatePassword(value) }))
    }
  }

  const handleBlur = (field) => {
    setTouched((prev) => ({ ...prev, [field]: true }))
    setFieldErrors((prev) => ({
      ...prev,
      email: field === 'email' ? validateEmail(email) : prev.email,
      password: field === 'password' ? validatePassword(password) : prev.password,
    }))
  }

  const handleSubmit = async (e) => {
    e.preventDefault()
    setError('')
    const nextErrors = {
      email: validateEmail(email),
      password: validatePassword(password),
    }
    setTouched({ email: true, password: true })
    setFieldErrors(nextErrors)
    if (nextErrors.email || nextErrors.password) {
      const focusId = nextErrors.email ? 'email' : 'password'
      requestAnimationFrame(() => document.getElementById(focusId)?.focus())
      return
    }

    setLoading(true)
    try {
      await login(email.trim(), password)
      try {
        sessionStorage.setItem('knowa.postLoginHandoff', '1')
      } catch {
        /* ignore */
      }
      navigate('/')
    } catch (err) {
      setError(err.message || 'Could not sign in')
      requestAnimationFrame(() => document.getElementById('email')?.focus())
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="min-h-screen grid lg:grid-cols-[1.05fr_0.95fr] bg-paper">
      <section className="relative hidden lg:flex flex-col justify-between p-12 xl:p-16 overflow-hidden border-r border-mist">
        <div
          className="pointer-events-none absolute inset-0"
          aria-hidden="true"
          style={{
            background:
              'radial-gradient(ellipse 90% 70% at 8% 18%, color-mix(in srgb, var(--teal) 28%, transparent), transparent 52%), radial-gradient(ellipse 70% 55% at 92% 88%, color-mix(in srgb, var(--coral) 16%, transparent), transparent 48%), linear-gradient(165deg, color-mix(in srgb, var(--surface) 70%, transparent), transparent 60%)',
          }}
        />
        <div
          className="pointer-events-none absolute inset-0 opacity-[0.35] auth-grid"
          aria-hidden="true"
        />

        <div className="relative auth-fade-in">
          <p className="text-[11px] font-semibold tracking-[0.18em] uppercase text-teal">
            Decision intelligence
          </p>
        </div>

        <div className="relative max-w-lg auth-fade-in" style={{ animationDelay: '80ms' }}>
          <h1 className="font-display text-6xl xl:text-7xl font-semibold tracking-tight leading-[0.9] text-ink">
            KNOWA
          </h1>
          <p className="mt-6 text-lg xl:text-xl text-[var(--muted)] leading-relaxed max-w-md">
            Predict with calibrated confidence. Explain every call. Hold the record.
          </p>

          <ol className="mt-12 space-y-0 border-l border-mist/80">
            {LOOP.map((step, i) => (
              <li
                key={step.label}
                className="relative pl-5 py-3 auth-fade-in"
                style={{ animationDelay: `${140 + i * 70}ms` }}
              >
                <span
                  className="absolute left-0 top-5 -translate-x-1/2 h-2 w-2 rounded-full bg-teal"
                  aria-hidden="true"
                />
                <div className="flex flex-wrap items-baseline gap-x-3 gap-y-0.5">
                  <span className="font-medium text-ink">{step.label}</span>
                  <span className="text-sm text-[var(--muted)]">{step.hint}</span>
                </div>
              </li>
            ))}
          </ol>
        </div>

        <p
          className="relative text-sm text-[var(--muted)] auth-fade-in"
          style={{ animationDelay: '420ms' }}
        >
          Accountable business decisions — not a black-box scoreboard.
        </p>
      </section>

      <section className="relative flex flex-col justify-center px-6 py-12 sm:px-12 lg:px-16">
        <div
          className="pointer-events-none absolute inset-0 lg:hidden opacity-80"
          aria-hidden="true"
          style={{
            background:
              'radial-gradient(ellipse 80% 40% at 50% 0%, color-mix(in srgb, var(--teal) 18%, transparent), transparent 55%)',
          }}
        />

        <div className="relative w-full max-w-md mx-auto animate-page-in">
          <div className="lg:hidden mb-10">
            <p className="font-display text-3xl font-semibold tracking-tight text-ink">KNOWA</p>
            <p className="mt-2 text-sm text-[var(--muted)]">Decide with clarity</p>
          </div>

          <h2 className="font-display text-2xl md:text-3xl font-semibold text-ink tracking-tight">
            Sign in
          </h2>
          <p className="mt-2 text-sm text-[var(--muted)] leading-relaxed">
            Continue to your workspace — cases, follow-ups, and what-ifs.
          </p>

          {error && (
            <div
              className="mt-6 text-sm text-ink border border-coral/40 bg-coral-soft px-4 py-3 rounded-control"
              role="alert"
            >
              {error}
            </div>
          )}

          <form onSubmit={handleSubmit} className="mt-8 space-y-5" noValidate>
            <div>
              <label htmlFor="email" className="block text-sm font-medium text-ink mb-1.5">
                Email
              </label>
              <input
                id="email"
                type="email"
                value={email}
                onChange={(e) => setEmailValue(e.target.value)}
                onBlur={() => handleBlur('email')}
                className={`input ${fieldErrors.email ? 'border-coral/50 focus:border-coral focus:ring-coral/25' : ''}`}
                placeholder="you@company.com"
                autoComplete="email"
                autoFocus
                disabled={loading}
                aria-invalid={Boolean(fieldErrors.email)}
                aria-describedby={fieldErrors.email ? 'email-error' : undefined}
              />
              {fieldErrors.email && (
                <p id="email-error" className="mt-1.5 text-xs text-coral" role="alert">
                  {fieldErrors.email}
                </p>
              )}
            </div>
            <div>
              <label htmlFor="password" className="block text-sm font-medium text-ink mb-1.5">
                Password
              </label>
              <div className="relative">
                <input
                  id="password"
                  type={showPassword ? 'text' : 'password'}
                  value={password}
                  onChange={(e) => setPasswordValue(e.target.value)}
                  onBlur={() => handleBlur('password')}
                  className={`input pr-20 ${fieldErrors.password ? 'border-coral/50 focus:border-coral focus:ring-coral/25' : ''}`}
                  placeholder="••••••••"
                  autoComplete="current-password"
                  disabled={loading}
                  aria-invalid={Boolean(fieldErrors.password)}
                  aria-describedby={fieldErrors.password ? 'password-error' : undefined}
                />
                <button
                  type="button"
                  className="absolute right-2 top-1/2 -translate-y-1/2 text-xs font-medium text-[var(--muted)] hover:text-teal px-2 py-1 rounded-control disabled:opacity-50"
                  onMouseDown={(e) => e.preventDefault()}
                  onClick={() => setShowPassword((v) => !v)}
                  disabled={loading}
                  aria-pressed={showPassword}
                  aria-label={showPassword ? 'Hide password' : 'Show password'}
                >
                  {showPassword ? 'Hide' : 'Show'}
                </button>
              </div>
              {fieldErrors.password && (
                <p id="password-error" className="mt-1.5 text-xs text-coral" role="alert">
                  {fieldErrors.password}
                </p>
              )}
            </div>
            <button
              type="submit"
              disabled={loading}
              className="btn-primary w-full py-3 mt-1 auth-cta"
            >
              {loading ? 'Signing in…' : 'Sign in'}
            </button>
          </form>

          <p className="mt-10 text-sm text-[var(--muted)]">
            No account?{' '}
            <Link to="/signup" className="text-teal font-medium hover:underline underline-offset-2">
              Create a workspace
            </Link>
          </p>
        </div>
      </section>
    </div>
  )
}
