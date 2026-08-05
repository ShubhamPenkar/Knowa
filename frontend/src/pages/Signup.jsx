import { useState } from 'react';
import { useNavigate, Link } from 'react-router-dom';
import { useAuth } from '../context/AuthContext';

export default function Signup() {
  const [formData, setFormData] = useState({
    organization_name: '',
    organization_slug: '',
    industry: 'saas',
    email: '',
    password: '',
    name: '',
  });
  const [error, setError] = useState('');
  const [loading, setLoading] = useState(false);
  const { signup } = useAuth();
  const navigate = useNavigate();

  const handleChange = (e) => {
    const { name, value } = e.target;
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
    }));
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    setError('');

    if (!formData.name.trim()) {
      setError('Please enter your name');
      return;
    }
    if (!formData.organization_name.trim()) {
      setError('Please enter your company name');
      return;
    }
    if (!formData.email.trim() || !formData.email.includes('@')) {
      setError('Please enter a valid email address');
      return;
    }
    if (formData.password.length < 6) {
      setError('Password must be at least 6 characters');
      return;
    }

    setLoading(true);
    try {
      await signup(formData);
      navigate('/');
    } catch (err) {
      setError(err.message);
    }
    setLoading(false);
  };

  return (
    <div className="min-h-screen grid lg:grid-cols-2 bg-paper">
      <section className="relative hidden lg:flex flex-col justify-between p-12 border-r border-mist">
        <p className="text-[11px] font-semibold tracking-[0.18em] uppercase text-teal">
          Get started
        </p>
        <div>
          <h1 className="font-display text-6xl font-semibold tracking-tight leading-[0.92] text-ink">
            KNOWA
          </h1>
          <p className="mt-6 max-w-sm text-lg text-muted leading-relaxed">
            Train on your data. See the interval. Act only when confidence allows.
          </p>
        </div>
        <p className="text-sm text-muted">One workspace for prediction and accountability</p>
      </section>

      <section className="flex flex-col justify-center px-6 py-12 sm:px-12 animate-page-in">
        <div className="lg:hidden mb-8">
          <p className="font-display text-3xl font-semibold tracking-tight text-ink">KNOWA</p>
        </div>
        <div className="w-full max-w-md mx-auto">
          <h2 className="font-display text-2xl font-semibold text-ink">Create workspace</h2>
          <p className="mt-1 text-sm text-muted">Organization + your account</p>

          {error && (
            <div className="mt-5 text-sm border border-coral/40 bg-coral-soft px-4 py-3 rounded-control" role="alert">
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
                  className="input"
                  placeholder={ph}
                />
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
                className="input"
                placeholder="you@company.com"
                autoComplete="email"
              />
            </div>
            <div>
              <label htmlFor="password" className="block text-sm font-medium text-ink mb-1">
                Password
              </label>
              <input
                id="password"
                name="password"
                type="password"
                value={formData.password}
                onChange={handleChange}
                className="input"
                placeholder="At least 6 characters"
                autoComplete="new-password"
              />
            </div>
            <button type="submit" disabled={loading} className="btn-primary w-full py-3 mt-2">
              {loading ? 'Creating…' : 'Create account'}
            </button>
          </form>

          <p className="mt-6 text-sm text-muted">
            Already set up?{' '}
            <Link to="/login" className="text-teal font-medium hover:underline">
              Sign in
            </Link>
          </p>
        </div>
      </section>
    </div>
  );
}
