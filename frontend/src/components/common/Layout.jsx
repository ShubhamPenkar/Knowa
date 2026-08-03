import React from 'react'
import { Outlet, Link, useLocation } from 'react-router-dom'
import { 
  Database,
  FolderKanban,
  BarChart3, 
  Brain,
  LogOut
} from 'lucide-react'
import { useAuth } from '../../context/AuthContext'

const navItems = [
  { path: '/', icon: FolderKanban, label: 'Projects' },
  { path: '/datasets', icon: Database, label: 'Datasets' },
  { path: '/analytics', icon: BarChart3, label: 'Analytics' },
]

function Layout() {
  const location = useLocation()
  const { user, organization, logout } = useAuth()

  return (
    <div className="min-h-screen flex">
      {/* Sidebar - Updated with brand colors */}
      <aside className="w-64 bg-brand-dark text-white flex flex-col">
        <div className="p-6">
          <div className="flex items-center gap-3">
            <Brain className="w-8 h-8 text-brand-teal" />
            <div>
              <h1 className="font-bold text-lg">Decision Intelligence</h1>
              <p className="text-xs text-gray-400">{organization?.name || 'Explainable AI'}</p>
            </div>
          </div>
        </div>
        
        <nav className="mt-2 flex-1">
          {navItems.map((item) => {
            const Icon = item.icon
            const isActive = location.pathname === item.path || 
              (item.path === '/' && location.pathname.startsWith('/projects'))
            
            return (
              <Link
                key={item.path}
                to={item.path}
                className={`flex items-center gap-3 px-6 py-3 text-sm font-medium transition-colors ${
                  isActive 
                    ? 'bg-brand-teal-dark text-white' 
                    : 'text-gray-300 hover:bg-gray-800 hover:text-white'
                }`}
              >
                <Icon className="w-5 h-5" />
                {item.label}
              </Link>
            )
          })}
        </nav>

        {/* User section */}
        <div className="p-4 border-t border-gray-800">
          <div className="flex items-center justify-between">
            <div className="truncate">
              <div className="text-sm font-medium text-white truncate">{user?.name}</div>
              <div className="text-xs text-gray-400 truncate">{user?.email}</div>
            </div>
            <button
              onClick={logout}
              className="p-2 text-gray-400 hover:text-white hover:bg-gray-800 rounded"
              title="Logout"
            >
              <LogOut className="w-4 h-4" />
            </button>
          </div>
        </div>
      </aside>

      {/* Main content - Updated with brand light background */}
      <main className="flex-1 overflow-auto bg-brand-light">
        <Outlet />
      </main>
    </div>
  )
}

export default Layout
