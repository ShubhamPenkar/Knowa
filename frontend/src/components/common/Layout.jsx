import React, { useCallback, useEffect, useState } from 'react'
import { Outlet, Link, useLocation } from 'react-router-dom'
import { useAuth } from '../../context/AuthContext'

/** Primary work tabs first; setup (Projects / Data) last. */
const navItems = [
  {
    path: '/',
    match: (p) => p === '/',
    label: 'Home',
    hint: "Knowa overview and today's pulse",
  },
  {
    path: '/cases',
    match: (p) => p.startsWith('/cases'),
    label: 'Cases',
    hint: 'Review who needs attention',
  },
  {
    path: '/follow-ups',
    match: (p) => p.startsWith('/follow-ups') || p.startsWith('/analytics'),
    label: 'Follow-ups',
    hint: 'Check-ins due across projects',
  },
  {
    path: '/whatif',
    match: (p) => p.startsWith('/whatif'),
    label: 'What-if',
    hint: 'Test scenarios before you act',
  },
  {
    path: '/monitoring',
    match: (p) => p.startsWith('/monitoring'),
    label: 'Monitoring',
    hint: 'Model quality and learning',
  },
  {
    path: '/projects',
    match: (p) => p === '/projects' || /^\/projects\/[^/]+$/.test(p),
    label: 'Projects',
    hint: 'Create and prepare models',
  },
  {
    path: '/datasets',
    match: (p) => p.startsWith('/datasets'),
    label: 'Data',
    hint: 'Upload datasets',
  },
]

function Layout() {
  const location = useLocation()
  const { user, organization, logout, token } = useAuth()
  const [mobileOpen, setMobileOpen] = useState(false)
  const [dueCount, setDueCount] = useState(0)

  const fetchDueBadge = useCallback(async () => {
    if (!token) {
      setDueCount(0)
      return
    }
    try {
      const res = await fetch('/api/projects/decisions/portfolio?limit=1', {
        headers: { Authorization: `Bearer ${token}` },
      })
      if (!res.ok) return
      const data = await res.json()
      const overdue = Number(data?.counts?.overdue || 0)
      const dueNow = Number(data?.counts?.due_now || 0)
      setDueCount(overdue + dueNow)
    } catch {
      /* badge is best-effort */
    }
  }, [token])

  useEffect(() => {
    fetchDueBadge()
  }, [fetchDueBadge, location.pathname])

  const Nav = ({ onNavigate }) => (
    <nav className="flex flex-col gap-0.5" aria-label="Primary">
      {navItems.map((item, idx) => {
        const isActive = item.match(location.pathname)
        const showBadge = item.path === '/follow-ups' && dueCount > 0
        const showDivider = idx === 5
        return (
          <React.Fragment key={item.path === '/' ? 'home' : item.path}>
            {showDivider && (
              <div
                className="my-2 mx-1 border-t border-mist"
                role="separator"
                aria-hidden="true"
              />
            )}
            <Link
              to={item.path}
              onClick={onNavigate}
              title={item.hint}
              className={`relative pl-3.5 pr-3 py-2.5 text-sm rounded-control transition-colors flex items-center justify-between gap-2 ${
                isActive
                  ? 'text-ink font-semibold bg-mist/50'
                  : 'text-muted font-medium hover:text-ink hover:bg-mist/30'
              }`}
            >
              {isActive && (
                <span
                  className="absolute left-0 top-1/2 -translate-y-1/2 h-6 w-[3px] rounded-sm bg-teal"
                  aria-hidden="true"
                />
              )}
              <span>{item.label}</span>
              {showBadge && (
                <span
                  className="badge shrink-0 bg-coral-soft border border-coral/30 text-coral tabular-nums"
                  title={`${dueCount} follow-up${dueCount === 1 ? '' : 's'} need attention`}
                >
                  {dueCount > 99 ? '99+' : dueCount}
                </span>
              )}
            </Link>
          </React.Fragment>
        )
      })}
    </nav>
  )

  return (
    <div className="min-h-screen flex bg-paper">
      <aside className="hidden lg:flex w-[15.5rem] shrink-0 flex-col border-r border-mist bg-paper relative z-10">
        <div className="px-5 pt-7 pb-6">
          <Link to="/" className="block group">
            <span className="font-display text-[1.65rem] font-semibold tracking-tight leading-none text-ink group-hover:text-teal transition-colors">
              KNOWA
            </span>
            <span className="block mt-2 text-[11px] text-muted tracking-wide truncate max-w-[11rem]">
              {organization?.name || 'Decide with clarity'}
            </span>
          </Link>
        </div>

        <div className="px-3 flex-1">
          <Nav />
        </div>

        <div className="m-3 mt-auto p-3 border-t border-mist">
          <div className="text-sm font-medium text-ink truncate">{user?.name}</div>
          <div className="text-xs text-muted truncate mt-0.5">{user?.email}</div>
          <button
            type="button"
            onClick={logout}
            className="mt-3 text-xs font-medium text-muted hover:text-teal transition-colors"
          >
            Sign out
          </button>
        </div>
      </aside>

      <div className="lg:hidden fixed inset-x-0 top-0 z-40 bg-paper/95 backdrop-blur-sm border-b border-mist">
        <div className="flex items-center justify-between px-4 h-14">
          <Link to="/" className="font-display text-lg font-semibold tracking-tight text-ink">
            KNOWA
          </Link>
          <button
            type="button"
            className="px-3 py-1.5 text-sm border border-mist text-ink rounded-control inline-flex items-center gap-2"
            onClick={() => setMobileOpen((v) => !v)}
            aria-expanded={mobileOpen}
            aria-controls="mobile-nav"
          >
            Menu
            {dueCount > 0 && (
              <span className="badge bg-coral-soft border border-coral/30 text-coral tabular-nums">
                {dueCount > 99 ? '99+' : dueCount}
              </span>
            )}
          </button>
        </div>
        {mobileOpen && (
          <div id="mobile-nav" className="px-3 pb-4 border-t border-mist bg-paper">
            <Nav onNavigate={() => setMobileOpen(false)} />
            <button
              type="button"
              onClick={logout}
              className="mt-3 ml-3 text-sm text-muted"
            >
              Sign out
            </button>
          </div>
        )}
      </div>

      <main className="relative flex-1 min-h-screen overflow-auto bg-paper pt-14 lg:pt-0">
        <div
          className="pointer-events-none absolute inset-0 shell-stage"
          aria-hidden="true"
        />
        <div
          className="pointer-events-none absolute inset-0 opacity-[0.22] stage-grid"
          aria-hidden="true"
        />
        <div className="relative z-[1] min-h-full">
          <Outlet />
        </div>
      </main>
    </div>
  )
}

export default Layout
