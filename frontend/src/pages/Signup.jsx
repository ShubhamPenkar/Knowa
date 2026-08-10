import { useState } from 'react'
import { useNavigate, Link } from 'react-router-dom'
import { useAuth } from '../context/AuthContext'
import {
  validateEmail,
  validatePassword,
  validateRequired,
} from '../lib/authValidation'

const EMPTY_ERRORS = {
  name: '',
  organization_name: '',
  email: '',
  password: '',
}

export default function Signup() {
  const [formData, setFormData] = useState({
    organization_name: '',
    organization_slug: '',
    industry: 'saas',
    email: '',
    password: '',
    name: '',
  })
  const [fieldErrors, setFieldErrors] = useState(EMPTY_ERRORS)
  const [touched, setTouched] = useState({})
  const [showPassword, setShowPassword] = useState(false)
  const [error, setError] = useState('')
  const [loading, setLoading] = useState(false)
  const { signup } = useAuth()
  const navigate = useNavigate()

  const validateField = (name, value) => {
    if (name === 'name') return validateRequired(value, 'your name')
    if (name === 'organization_name') return validateRequired(value, 'your company name')
    if (name === 'email') return validateEmail(value)
    if (name === 'password') return validatePassword(value, { minLength: 6 })
    return ''
  }

  const handleChange = (e) => {
    const { name, value } = e.target
    if (error) setError('')
    setFormData((prev) => ({
      ...prev,
      [name]: value,
      ...(name === 'organization_name'
        ? {
            organization_slug: value
              .toLowerCase()
              .replace(/[^a-z0-9]+/g, '-')
              .replace(/^-+|-+$/g, '')
              .substring(0, 50),
          }
        : {}),
    }))
    if (touched[name]) {
      setFieldErrors((prev) => ({ ...prev, [name]: validateField(name, value) }))
    }
  }

  const handleBlur = (name) => {
    setTouched((prev) => ({ ...prev, [name]: true }))
    setFieldErrors((prev) => ({
      ...prev,
      [name]: validateField(name, formData[name]),
    }))
  }

  const handleSubmit = async (e) => {
    e.preventDefault()
    setError('')

    const nextErrors = {
      name: validateField('name', formData.name),
      organization_name: validateField('organization_name', formData.organization_name),
      email: validateField('email', formData.email),
      password: validateField('password', formData.password),
    }
    if (!nextErrors.organization_name && !formData.organization_slug) {
      nextErrors.organization_name =
        'Company name needs letters or numbers for a workspace URL'
    }
    setTouched({
      name: true,
      organization_name: true,
      email: true,
      password: true,
    })
    setFieldErrors(nextErrors)

    const firstInvalid = ['name', 'organization_name', 'email', 'password'].find(
      (k) => nextErrors[k]
    )
    if (firstInvalid) {
      requestAnimationFrame(() => document.getElementById(firstInvalid)?.focus())
      return
    }

    setLoading(true)
    try {
      await signup({
        ...formData,
        name: formData.name.trim(),
        organization_name: formData.organization_name.trim(),
        email: formData.email.trim(),
      })
      try {
        sessionStorage.setItem('knowa.postLoginHandoff', '1')
      } catch {
        /* ignore */
      }
      navigate('/')
    } catch (err) {
      setError(err.message || 'Could not create account')
    } finally {
      setLoading(false)
    }
  }

  const fieldClass = (name) =>
    `input ${fieldErrors[name] ? 'border-coral/50 focus:border-coral focus:ring-coral/25' : ''}`

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
            Get started
          </p>
        </div>

        <div className="relative max-w-lg auth-fade-in" style={{ animationDelay: '80ms' }}>
          <h1 className="font-display text-6xl xl:text-7xl font-semibold tracking-tight leading-[0.9] text-ink">
            KNOWA
          </h1>
          <p className="mt-6 text-lg xl:text-xl text-[var(--muted)] leading-relaxed max-w-md">
            Train on your data. See the interval. Act only when confidence allows.
          </p>
          <p className="mt-8 text-sm text-[var(--muted)] leading-relaxed max-w-sm">
            One workspace for prediction, explanation, and accountable follow-ups.
          </p>
        </div>

        <p
          className="relative text-sm text-[var(--muted)] auth-fade-in"
          style={{ animationDelay: '280ms' }}
        >
          Takes a minute — then you&apos;re into the decision loop.
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
          <div className="lg:hidden mb-8">
            <p className="font-display text-3xl font-semibold tracking-tight text-ink">KNOWA</p>
            <p className="mt-2 text-sm text-[var(--muted)]">Create your workspace</p>
          </div>
          <h2 className="font-display text-2xl md:text-3xl font-semibold text-ink tracking-tight">
            Create workspace
          </h2>
          <p className="mt-2 text-sm text-[var(--muted)]">Organization + your account</p>

          {error && (
            <div
              className="mt-5 text-sm border border-coral/40 bg-coral-soft px-4 py-3 rounded-control"
              role="alert"
            >
              {error}
            </div>
          )}

          <form onSubmit={handleSubmit} className="mt-6 space-y-3.5" noValidate>
            {[
              ['name', 'Your name', 'text', 'Jordan Lee'],
              ['organization_name', 'Company name', 'text', 'Acme Inc'],
            ].map(([name, label, type, ph]) => (
              <div key={name}>
                <label htmlFor={name} className="block text-sm font-medium text-ink mb-1">
                  {label}
                </label>
                <input
                  id={name}
                  name={name}
                  type={type}
                  value={formData[name]}
                  onChange={handleChange}
                  onBlur={() => handleBlur(name)}
                  className={fieldClass(name)}
                  placeholder={ph}
                  disabled={loading}
                  aria-invalid={Boolean(fieldErrors[name])}
                  aria-describedby={fieldErrors[name] ? `${name}-error` : undefined}
                />
                {fieldErrors[name] && (
                  <p id={`${name}-error`} className="mt-1 text-xs text-coral" role="alert">
                    {fieldErrors[name]}
                  </p>
                )}
              </div>
            ))}
            <div>
              <label htmlFor="industry" className="block text-sm font-medium text-ink mb-1">
                Industry
              </label>
              <select
                id="industry"
                name="industry"
                value={formData.industry}
                onChange={handleChange}
                className="input"
                disabled={loading}
              >
                <option value="saas">SaaS / Software</option>
                <option value="ecommerce">E-commerce / Retail</option>
                <option value="finance">Finance / Banking</option>
                <option value="healthcare">Healthcare</option>
                <option value="other">Other</option>
              </select>
            </div>
            <div>
              <label htmlFor="email" className="block text-sm font-medium text-ink mb-1">
                Work email
              </label>
              <input
                id="email"
                name="email"
                type="email"
                value={formData.email}
                onChange={handleChange}
                onBlur={() => handleBlur('email')}
                className={fieldClass('email')}
                placeholder="you@company.com"
                autoComplete="email"
                disabled={loading}
                aria-invalid={Boolean(fieldErrors.email)}
                aria-describedby={fieldErrors.email ? 'email-error' : undefined}
              />
              {fieldErrors.email && (
                <p id="email-error" className="mt-1 text-xs text-coral" role="alert">
                  {fieldErrors.email}
                </p>
              )}
            </div>
            <div>
              <label htmlFor="password" className="block text-sm font-medium text-ink mb-1">
                Password
              </label>
              <div className="relative">
                <input
                  id="password"
                  name="password"
                  type={showPassword ? 'text' : 'password'}
                  value={formData.password}
                  onChange={handleChange}
                  onBlur={() => handleBlur('password')}
                  className={`${fieldClass('password')} pr-20`}
                  placeholder="At least 6 characters"
                  autoComplete="new-password"
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
                <p id="password-error" className="mt-1 text-xs text-coral" role="alert">
                  {fieldErrors.password}
                </p>
              )}
            </div>
            <button
              type="submit"
              disabled={loading}
              className="btn-primary w-full py-3 mt-2 auth-cta"
            >
              {loading ? 'Creating…' : 'Create account'}
            </button>
          </form>

          <p className="mt-8 text-sm text-[var(--muted)]">
            Already set up?{' '}
            <Link to="/login" className="text-teal font-medium hover:underline underline-offset-2">
              Sign in
            </Link>
          </p>
        </div>
      </section>
    </div>
  )
}
