import React, { useState } from 'react'
import { Outlet, Link, useLocation } from 'react-router-dom'
import { useAuth } from '../../context/AuthContext'

const navItems = [
  { path: '/', match: (p) => p === '/' || p.startsWith('/projects'), label: 'Projects' },
  { path: '/datasets', match: (p) => p.startsWith('/datasets'), label: 'Data' },
  { path: '/analytics', match: (p) => p.startsWith('/analytics'), label: 'Priorities' },
]

function Layout() {
  const location = useLocation()
  const { user, organization, logout } = useAuth()
  const [mobileOpen, setMobileOpen] = useState(false)

  const Nav = ({ onNavigate }) => (
    <nav className="flex flex-col gap-0.5" aria-label="Primary">
      {navItems.map((item) => {
        const isActive = item.match(location.pathname)
        return (
          <Link
            key={item.path}
            to={item.path}
            onClick={onNavigate}
            className={`relative px-3 py-2.5 text-sm font-medium rounded-control transition-colors ${
              isActive
                ? 'text-ink bg-mist/60'
                : 'text-muted hover:text-ink hover:bg-mist/30'
            }`}
          >
            {isActive && (
              <span
                className="absolute left-0 top-1/2 -translate-y-1/2 h-5 w-0.5 bg-teal"
                aria-hidden="true"
              />
            )}
            {item.label}
          </Link>
        )
      })}
    </nav>
  )

  return (
    <div className="min-h-screen flex bg-paper">
      <aside className="hidden lg:flex w-[15.5rem] shrink-0 flex-col border-r border-mist bg-paper">
        <div className="px-5 pt-7 pb-6">
          <Link to="/" className="block">
            <span className="font-display text-[1.65rem] font-semibold tracking-tight leading-none text-ink">
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

      <div className="lg:hidden fixed inset-x-0 top-0 z-40 bg-paper border-b border-mist">
        <div className="flex items-center justify-between px-4 h-14">
          <Link to="/" className="font-display text-lg font-semibold tracking-tight text-ink">
            KNOWA
          </Link>
          <button
            type="button"
            className="px-3 py-1.5 text-sm border border-mist text-ink rounded-control"
            onClick={() => setMobileOpen((v) => !v)}
            aria-expanded={mobileOpen}
            aria-controls="mobile-nav"
          >
            Menu
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

      <main className="flex-1 min-h-screen overflow-auto bg-paper pt-14 lg:pt-0">
        <Outlet />
      </main>
    </div>
  )
}

export default Layout
