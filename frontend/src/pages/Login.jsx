import { useState } from 'react';
import { useNavigate, Link } from 'react-router-dom';
import { useAuth } from '../context/AuthContext';

export default function Login() {
  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');
  const [error, setError] = useState('');
  const [loading, setLoading] = useState(false);
  const { login } = useAuth();
  const navigate = useNavigate();

  const handleSubmit = async (e) => {
    e.preventDefault();
    setError('');
    setLoading(true);
    try {
      await login(email, password);
      navigate('/cases');
    } catch (err) {
      setError(err.message);
    }
    setLoading(false);
  };

  return (
    <div className="min-h-screen grid lg:grid-cols-2 bg-paper">
      <section className="relative hidden lg:flex flex-col justify-between p-12 border-r border-mist">
        <p className="text-[11px] font-semibold tracking-[0.18em] uppercase text-teal">
          Explainable decisions
        </p>
        <div>
          <h1 className="font-display text-6xl xl:text-7xl font-semibold tracking-tight leading-[0.92] text-ink">
            KNOWA
          </h1>
          <p className="mt-6 max-w-sm text-lg text-muted leading-relaxed">
            Predict with calibrated confidence. Explain every call. Hold the record.
          </p>
          <div className="mt-10 flex gap-3 text-xs font-medium tracking-wide">
            <span className="px-2 py-1 border border-coral/40 text-coral rounded-control">Risk</span>
            <span className="text-muted self-center">→</span>
            <span className="px-2 py-1 border border-teal/40 text-teal rounded-control">Resolution</span>
          </div>
        </div>
        <p className="text-sm text-muted">Accountable business decision intelligence</p>
      </section>

      <section className="flex flex-col justify-center px-6 py-12 sm:px-12 animate-page-in">
        <div className="lg:hidden mb-10">
          <p className="font-display text-3xl font-semibold tracking-tight text-ink">KNOWA</p>
        </div>
        <div className="w-full max-w-md mx-auto">
          <h2 className="font-display text-2xl font-semibold text-ink">Sign in</h2>
          <p className="mt-1 text-sm text-muted">Continue to your workspace</p>

          {error && (
            <div
              className="mt-5 text-sm text-ink border border-coral/40 bg-coral-soft px-4 py-3 rounded-control"
              role="alert"
            >
              {error}
            </div>
          )}

          <form onSubmit={handleSubmit} className="mt-8 space-y-4" noValidate>
            <div>
              <label htmlFor="email" className="block text-sm font-medium text-ink mb-1.5">
                Email
              </label>
              <input
                id="email"
                type="email"
                value={email}
                onChange={(e) => setEmail(e.target.value)}
                className="input"
                placeholder="you@company.com"
                autoComplete="email"
                required
              />
            </div>
            <div>
              <label htmlFor="password" className="block text-sm font-medium text-ink mb-1.5">
                Password
              </label>
              <input
                id="password"
                type="password"
                value={password}
                onChange={(e) => setPassword(e.target.value)}
                className="input"
                autoComplete="current-password"
                required
              />
            </div>
            <button type="submit" disabled={loading} className="btn-primary w-full py-3 mt-2">
              {loading ? 'Signing in…' : 'Sign in'}
            </button>
          </form>

          <p className="mt-8 text-sm text-muted">
            No account?{' '}
            <Link to="/signup" className="text-teal font-medium hover:underline">
              Create one
            </Link>
          </p>
        </div>
      </section>
    </div>
  );
}
