import React, { useCallback, useEffect, useState } from 'react'
import { Outlet, Link, useLocation } from 'react-router-dom'
import {
  Home,
  Briefcase,
  Scale,
  FlaskConical,
  Database,
  Activity,
  FolderKanban,
  Settings,
  LogOut,
  Menu,
  X,
  Sparkles,
  Code,
} from 'lucide-react'
import { useAuth } from '../../context/AuthContext'
import { useViewMode } from '../../context/ViewModeContext'

const getNavItems = (plainEnglish) => [
  {
    path: '/',
    match: (p) => p === '/',
    label: 'Home',
    hint: plainEnglish ? 'Business overview and daily priority decisions' : 'Executive command center and daily decision pulse',
    icon: Home,
  },
  {
    path: '/cases',
    match: (p) => p.startsWith('/cases'),
    label: plainEnglish ? 'Accounts & Cases' : 'Cases Queue',
    hint: plainEnglish ? 'Prioritized customer accounts needing action' : 'Prioritized accounts and individual decisions',
    icon: Briefcase,
  },
  {
    path: '/follow-ups',
    match: (p) => p.startsWith('/follow-ups') || p.startsWith('/analytics'),
    label: plainEnglish ? 'Action History' : 'Decision Ledger',
    hint: plainEnglish ? 'Committed business actions and follow-up results' : 'Committed actions, check-ins, and portfolio ROI',
    icon: Scale,
  },
  {
    path: '/whatif',
    match: (p) => p.startsWith('/whatif'),
    label: plainEnglish ? 'Scenario Simulator' : 'What-If Simulation',
    hint: plainEnglish ? 'Test business changes before taking action' : 'Test scenarios and intervention levers before acting',
    icon: FlaskConical,
  },
  {
    path: '/datasets',
    match: (p) => p.startsWith('/datasets'),
    label: plainEnglish ? 'Data & Insights' : 'Data Foundation',
    hint: plainEnglish ? 'Your tables, connections, and automated discoveries' : 'Catalog, dataset connections, and dynamic findings',
    icon: Database,
  },
  {
    path: '/monitoring',
    match: (p) => p.startsWith('/monitoring'),
    label: plainEnglish ? 'AI Accuracy & Drift' : 'Model Health',
    hint: plainEnglish ? 'Prediction certainty and accuracy over time' : 'Prediction reliability, accuracy, and operational drift',
    icon: Activity,
  },
  {
    path: '/projects',
    match: (p) => p === '/projects' || /^\/projects\/[^/]+$/.test(p),
    label: 'Projects',
    hint: plainEnglish ? 'Business objectives and prediction goals' : 'Business objectives and model pipelines',
    icon: FolderKanban,
  },
]

function Layout() {
  const location = useLocation()
  const { user, organization, logout, token } = useAuth()
  const { plainEnglish, setPlainEnglish, toggleMode } = useViewMode()
  const [mobileOpen, setMobileOpen] = useState(false)
  const [dueCount, setDueCount] = useState(0)

  const navItems = getNavItems(plainEnglish)

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
        const showDivider = idx === 4
        const Icon = item.icon
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
              className={`group relative px-3 py-2 text-sm rounded-control transition-all flex items-center justify-between gap-2.5 ${
                isActive
                  ? 'text-ink font-semibold bg-mist/50 shadow-sm'
                  : 'text-muted font-medium hover:text-ink hover:bg-mist/30'
              }`}
            >
              {isActive && (
                <span
                  className="absolute left-0 top-1/2 -translate-y-1/2 h-6 w-[3px] rounded-sm bg-teal shadow-[0_0_8px_rgba(0,200,180,0.5)]"
                  aria-hidden="true"
                />
              )}
              <div className="flex items-center gap-2.5 min-w-0">
                {Icon && (
                  <Icon
                    size={17}
                    className={`shrink-0 transition-colors ${
                      isActive ? 'text-teal' : 'text-muted group-hover:text-ink'
                    }`}
                  />
                )}
                <span className="truncate">{item.label}</span>
              </div>
              {showBadge && (
                <span
                  className="badge shrink-0 bg-coral-soft border border-coral/30 text-coral tabular-nums font-semibold"
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

  const userInitials = (user?.name || user?.email || 'U')
    .slice(0, 2)
    .toUpperCase()

  return (
    <div className="min-h-screen flex bg-paper">
      <aside className="hidden lg:flex w-[16rem] shrink-0 flex-col border-r border-mist bg-paper relative z-10">
        <div className="px-5 pt-7 pb-4">
          <Link to="/" className="block group">
            <div className="flex items-center gap-2">
              <span className="font-display text-[1.65rem] font-semibold tracking-tight leading-none text-ink group-hover:text-teal transition-colors">
                KNOWA
              </span>
              <span className="h-1.5 w-1.5 rounded-full bg-teal shadow-[0_0_6px_rgba(0,200,180,0.8)]" />
            </div>
            <span className="block mt-2 text-[11px] text-muted tracking-wide truncate max-w-[11rem]">
              {organization?.name || 'Decide with clarity'}
            </span>
          </Link>

          {/* Global Mode Switcher in Sidebar */}
          <div className="mt-4 p-2 bg-surface/90 border border-mist rounded-control">
            <div className="flex items-center justify-between px-1 mb-1.5">
              <span className="text-[10px] uppercase font-semibold tracking-wider text-[var(--muted)]">View Language</span>
              <span className="text-[10px] text-teal font-medium">{plainEnglish ? 'Plain' : 'Dev'}</span>
            </div>
            <div className="grid grid-cols-2 gap-1 bg-paper/80 p-0.5 rounded border border-mist text-xs">
              <button
                type="button"
                onClick={() => setPlainEnglish(true)}
                className={`py-1 px-1.5 rounded text-[11px] font-medium transition-all flex items-center justify-center gap-1 ${
                  plainEnglish ? 'bg-teal text-paper font-semibold shadow-xs' : 'text-muted hover:text-ink'
                }`}
                title="Plain English mode for business users"
              >
                <Sparkles size={11} />
                <span>Plain</span>
              </button>
              <button
                type="button"
                onClick={() => setPlainEnglish(false)}
                className={`py-1 px-1.5 rounded text-[11px] font-medium transition-all flex items-center justify-center gap-1 ${
                  !plainEnglish ? 'bg-teal text-paper font-semibold shadow-xs' : 'text-muted hover:text-ink'
                }`}
                title="Developer mode with raw database & ML terms"
              >
                <Code size={11} />
                <span>Dev</span>
              </button>
            </div>
          </div>
        </div>

        <div className="px-3 flex-1 overflow-y-auto">
          <Nav />
        </div>

        <div className="m-3 mt-auto p-3 border border-mist/70 rounded-control bg-surface/50">
          <Link
            to="/settings"
            className="flex items-center gap-2.5 rounded-control hover:bg-mist/30 transition-colors p-1"
            title="Open settings"
          >
            <div className="w-8 h-8 rounded-full bg-teal-soft border border-teal/40 flex items-center justify-center text-xs font-bold text-teal shrink-0">
              {userInitials}
            </div>
            <div className="flex-1 min-w-0">
              <div className="text-xs font-semibold text-ink truncate leading-tight">{user?.name || 'User'}</div>
              <div className="text-[11px] text-muted truncate">{user?.email}</div>
            </div>
          </Link>
          <div className="mt-2 pt-2 border-t border-mist/60 flex items-center justify-between text-xs">
            <Link
              to="/settings"
              className="inline-flex items-center gap-1 text-[11px] font-medium text-muted hover:text-teal transition-colors"
            >
              <Settings size={12} />
              Settings
            </Link>
            <button
              type="button"
              onClick={logout}
              className="inline-flex items-center gap-1 text-[11px] font-medium text-muted hover:text-coral transition-colors"
            >
              <LogOut size={12} />
              Sign out
            </button>
          </div>
        </div>
      </aside>

      {/* Mobile Top Header */}
      <div className="lg:hidden fixed inset-x-0 top-0 z-40 bg-paper/95 backdrop-blur-sm border-b border-mist">
        <div className="flex items-center justify-between px-4 h-14">
          <Link to="/" className="flex items-center gap-2 font-display text-lg font-semibold tracking-tight text-ink">
            <span>KNOWA</span>
            <span className="h-1.5 w-1.5 rounded-full bg-teal shadow-[0_0_6px_rgba(0,200,180,0.8)]" />
          </Link>
          
          <div className="flex items-center gap-2">
            {/* Quick Toggle Button on Mobile */}
            <button
              type="button"
              onClick={toggleMode}
              className="p-1.5 rounded bg-surface border border-mist text-xs text-teal font-medium flex items-center gap-1"
              title="Toggle Plain English vs Developer mode"
            >
              {plainEnglish ? <Sparkles size={13} /> : <Code size={13} />}
              <span className="text-[11px]">{plainEnglish ? 'Plain' : 'Dev'}</span>
            </button>

            <button
              type="button"
              className="px-3 py-1.5 text-sm border border-mist text-ink rounded-control inline-flex items-center gap-2"
              onClick={() => setMobileOpen((v) => !v)}
              aria-expanded={mobileOpen}
              aria-controls="mobile-nav"
            >
              {mobileOpen ? <X size={16} /> : <Menu size={16} />}
              <span>Menu</span>
              {dueCount > 0 && (
                <span className="badge bg-coral-soft border border-coral/30 text-coral tabular-nums">
                  {dueCount > 99 ? '99+' : dueCount}
                </span>
              )}
            </button>
          </div>
        </div>
        {mobileOpen && (
          <div id="mobile-nav" className="px-3 pb-4 border-t border-mist bg-paper">
            <Nav onNavigate={() => setMobileOpen(false)} />
            <div className="mt-3 pt-3 border-t border-mist/60 flex items-center justify-between px-2">
              <Link
                to="/settings"
                onClick={() => setMobileOpen(false)}
                className="inline-flex items-center gap-1.5 text-sm text-teal font-medium"
              >
                <Settings size={14} />
                Settings
              </Link>
              <button
                type="button"
                onClick={logout}
                className="inline-flex items-center gap-1.5 text-sm text-muted hover:text-coral"
              >
                <LogOut size={14} />
                Sign out
              </button>
            </div>
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
          {/* Global Sticky Topbar with Mode Toggle */}
          <header className="sticky top-0 z-30 flex items-center justify-between px-6 py-2.5 bg-paper/90 backdrop-blur-md border-b border-mist/80">
            <div className="flex items-center gap-2">
              <span className="text-xs text-[var(--muted)] flex items-center gap-1.5 font-medium">
                {plainEnglish ? (
                  <>
                    <Sparkles size={13} className="text-teal" />
                    <span className="text-ink/90">Plain English Mode</span>
                    <span className="text-[var(--muted)] hidden sm:inline">· Clear business language active</span>
                  </>
                ) : (
                  <>
                    <Code size={13} className="text-teal" />
                    <span className="text-ink/90">Developer Terms Mode</span>
                    <span className="text-[var(--muted)] hidden sm:inline">· Showing SQL joins, keys & ML internals</span>
                  </>
                )}
              </span>
            </div>

            <div className="flex items-center gap-2 sm:gap-3">
              <span className="text-[11px] text-[var(--muted)] font-medium hidden sm:inline">View Language:</span>
              <div className="flex items-center bg-surface border border-mist p-0.5 rounded-control shadow-xs">
                <button
                  type="button"
                  onClick={() => setPlainEnglish(true)}
                  className={`px-3 py-1 text-xs rounded transition-all flex items-center gap-1.5 font-medium ${
                    plainEnglish
                      ? 'bg-teal text-paper font-semibold shadow-xs'
                      : 'text-[var(--muted)] hover:text-ink'
                  }`}
                  title="Plain English mode replaces database and ML jargon with clear business terminology"
                >
                  <Sparkles size={12} className={plainEnglish ? 'text-paper' : 'text-teal'} />
                  <span>Plain English</span>
                </button>
                <button
                  type="button"
                  onClick={() => setPlainEnglish(false)}
                  className={`px-3 py-1 text-xs rounded transition-all flex items-center gap-1.5 font-medium ${
                    !plainEnglish
                      ? 'bg-teal text-paper font-semibold shadow-xs'
                      : 'text-[var(--muted)] hover:text-ink'
                  }`}
                  title="Developer mode displays raw database keys, SQL join types, and ML statistical internals"
                >
                  <Code size={12} />
                  <span>Developer Terms</span>
                </button>
              </div>
            </div>
          </header>

          <Outlet />
        </div>
      </main>
    </div>
  )
}

export default Layout
