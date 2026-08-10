import { useEffect, useMemo, useState } from 'react'
import { Link } from 'react-router-dom'
import { useAuth } from '../context/AuthContext'
import OrgHealthStrip from '../components/common/OrgHealthStrip'
import { getLastProjectId } from '../lib/lastProject'

const FLOW = [
  {
    n: '01',
    title: 'Data & project',
    body: 'Upload a dataset, describe the decision, and prepare a model.',
    to: '/projects',
    cta: 'Projects',
  },
  {
    n: '02',
    title: 'Cases',
    body: 'Open people or accounts, see why the score moved, and pick an action.',
    to: '/cases',
    cta: 'Review cases',
  },
  {
    n: '03',
    title: 'Follow-ups',
    body: 'Commit an action, assign an owner, and check in when it comes due.',
    to: '/follow-ups',
    cta: 'Follow-ups',
  },
  {
    n: '04',
    title: 'What-if & learn',
    body: 'Test levers before you spend. Outcomes reshape rankings over time.',
    to: '/whatif',
    cta: 'What-if',
  },
]

const CAPABILITIES = [
  {
    title: 'Explainable scores',
    body: 'Every prediction comes with drivers, business language, and trust signals — including when not to act.',
  },
  {
    title: 'Ranked actions',
    body: 'Recommendations weigh impact vs cost, then land on a durable follow-up you can assign and audit.',
  },
  {
    title: 'Portfolio intelligence',
    body: 'See ROI by action, capacity pressure, and overdue check-ins across the org — not just one case.',
  },
]

/**
 * In-app home — orients users on the Knowa decision loop before deep work tabs.
 */
export default function Home() {
  const { token, organization, user } = useAuth()
  const [projects, setProjects] = useState([])
  const [loading, setLoading] = useState(true)

  useEffect(() => {
    if (!token) {
      setLoading(false)
      return
    }
    let cancelled = false
    ;(async () => {
      try {
        const res = await fetch('/api/projects', {
          headers: { Authorization: `Bearer ${token}` },
        })
        if (res.ok && !cancelled) {
          setProjects(await res.json())
        }
      } catch {
        /* ignore */
      } finally {
        if (!cancelled) setLoading(false)
      }
    })()
    return () => {
      cancelled = true
    }
  }, [token])

  const readyProjects = useMemo(
    () => projects.filter((p) => p.status === 'ready' || p.status === 'trained'),
    [projects]
  )

  const lastId = getLastProjectId()
  const primaryProjectId =
    (lastId && readyProjects.some((p) => p.id === lastId) && lastId) ||
    readyProjects[0]?.id ||
    null

  const hasReady = readyProjects.length > 0
  const greetingName = user?.name?.split(' ')[0]

  return (
    <div className="page">
      <header className="relative overflow-hidden border border-mist mb-10">
        <div
          className="pointer-events-none absolute inset-0 opacity-90"
          aria-hidden="true"
          style={{
            background:
              'radial-gradient(ellipse 80% 60% at 12% 20%, color-mix(in srgb, var(--teal) 22%, transparent), transparent 55%), radial-gradient(ellipse 70% 50% at 88% 10%, color-mix(in srgb, var(--coral) 14%, transparent), transparent 50%), linear-gradient(180deg, color-mix(in srgb, var(--surface) 80%, transparent), transparent)',
          }}
        />
        <div className="relative px-6 py-10 md:px-10 md:py-14">
          <p className="page-kicker">Home</p>
          <h1 className="font-display text-4xl md:text-5xl font-semibold text-ink tracking-tight max-w-2xl leading-[1.05]">
            Knowa
          </h1>
          <p className="mt-4 text-base md:text-lg text-[var(--muted)] max-w-xl leading-relaxed">
            {greetingName ? `Welcome back, ${greetingName}. ` : ''}
            Decide with clarity — predict outcomes, see why, act with ranked moves, then learn from
            what happened.
          </p>
          <div className="mt-8 flex flex-wrap gap-3">
            {hasReady ? (
              <>
                <Link
                  to={primaryProjectId ? `/cases?project=${primaryProjectId}` : '/cases'}
                  className="btn-primary"
                >
                  Continue reviewing
                </Link>
                <Link to="/follow-ups" className="btn-secondary">
                  Open follow-ups
                </Link>
              </>
            ) : (
              <>
                <Link to="/projects" className="btn-primary">
                  Set up a project
                </Link>
                <Link to="/datasets" className="btn-secondary">
                  Upload data
                </Link>
              </>
            )}
          </div>
          {!loading && organization?.name && (
            <p className="mt-6 text-xs text-[var(--muted)]">
              Workspace · {organization.name}
              {hasReady
                ? ` · ${readyProjects.length} ready project${readyProjects.length === 1 ? '' : 's'}`
                : ' · no ready projects yet'}
            </p>
          )}
        </div>
      </header>

      <OrgHealthStrip projectId={primaryProjectId} />

      <section className="mb-12" aria-labelledby="flow-heading">
        <h2 id="flow-heading" className="font-display text-xl font-semibold text-ink">
          How Knowa works
        </h2>
        <p className="text-sm text-[var(--muted)] mt-2 max-w-xl">
          A closed loop — not a one-shot churn dashboard.
        </p>
        <ol className="mt-6 border border-mist divide-y divide-mist">
          {FLOW.map((step, i) => (
            <li
              key={step.n}
              className="px-5 py-5 md:px-6 flex flex-col md:flex-row md:items-center gap-4 md:gap-8 home-flow-row"
              style={{ animationDelay: `${80 + i * 70}ms` }}
            >
              <span className="font-display text-sm text-teal tabular-nums shrink-0 w-8">
                {step.n}
              </span>
              <div className="min-w-0 flex-1">
                <h3 className="font-medium text-ink">{step.title}</h3>
                <p className="text-sm text-[var(--muted)] mt-1 leading-relaxed">{step.body}</p>
              </div>
              <Link to={step.to} className="btn-ghost text-sm shrink-0 self-start md:self-center">
                {step.cta} →
              </Link>
            </li>
          ))}
        </ol>
      </section>

      <section className="mb-10" aria-labelledby="cap-heading">
        <h2 id="cap-heading" className="font-display text-xl font-semibold text-ink">
          What you can do here
        </h2>
        <p className="text-sm text-[var(--muted)] mt-2 max-w-xl">
          Built for operators who need accountable decisions — not raw model dumps.
        </p>
        <ul className="mt-6 grid md:grid-cols-3 gap-px bg-mist border border-mist">
          {CAPABILITIES.map((cap) => (
            <li key={cap.title} className="bg-paper px-5 py-6">
              <h3 className="font-medium text-ink">{cap.title}</h3>
              <p className="text-sm text-[var(--muted)] mt-2 leading-relaxed">{cap.body}</p>
            </li>
          ))}
        </ul>
      </section>

      <section className="border border-mist px-5 py-6 flex flex-wrap items-center justify-between gap-4">
        <div>
          <h2 className="font-display text-lg font-semibold text-ink">Ready to dig in?</h2>
          <p className="text-sm text-[var(--muted)] mt-1">
            {hasReady
              ? 'Jump back into cases, or check Monitoring when a guide looks rough.'
              : 'Start with data, then describe the decision you want to score.'}
          </p>
        </div>
        <div className="flex flex-wrap gap-2">
          <Link
            to={hasReady ? (primaryProjectId ? `/cases?project=${primaryProjectId}` : '/cases') : '/datasets'}
            className="btn-primary"
          >
            {hasReady ? 'Open cases' : 'Go to data'}
          </Link>
          <Link to="/monitoring" className="btn-ghost text-sm">
            Monitoring
          </Link>
        </div>
      </section>
    </div>
  )
}
