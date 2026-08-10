import { useEffect, useState } from 'react'
import { Link, Navigate, useNavigate } from 'react-router-dom'
import { useAuth } from '../context/AuthContext'
import ProjectPicker from '../components/common/ProjectPicker'
import Spinner from '../components/common/Spinner'
import { getLastProjectId, setLastProjectId } from '../lib/lastProject'

/**
 * Top-level What-if entry: pick a ready project, then open the scenario workspace.
 */
export default function WhatIfEntry() {
  const { token } = useAuth()
  const navigate = useNavigate()
  const [ready, setReady] = useState([])
  const [loading, setLoading] = useState(true)
  const [selected, setSelected] = useState('')
  const [autoNav, setAutoNav] = useState(null)

  useEffect(() => {
    if (!token) return
    ;(async () => {
      try {
        const res = await fetch('/api/projects', {
          headers: { Authorization: `Bearer ${token}` },
        })
        if (res.ok) {
          const data = await res.json()
          const list = data.filter((p) => p.status === 'ready' || p.status === 'trained')
          setReady(list)
          const last = getLastProjectId()
          const pick = list.find((p) => p.id === last)?.id || list[0]?.id || ''
          setSelected(pick)
          // One ready project → go straight in
          if (list.length === 1) setAutoNav(list[0].id)
        }
      } catch (err) {
        console.error(err)
      }
      setLoading(false)
    })()
  }, [token])

  if (loading) {
    return (
      <div className="page flex justify-center items-center min-h-[40vh]">
        <Spinner />
      </div>
    )
  }

  if (autoNav) {
    return <Navigate to={`/whatif/${autoNav}`} replace />
  }

  if (ready.length === 0) {
    return (
      <div className="page">
        <p className="page-kicker">Scenarios</p>
        <h1 className="page-title">What-if</h1>
        <div className="empty-state mt-8">
          <h3 className="font-display text-xl font-semibold text-ink mb-2">No ready projects</h3>
          <p className="text-sm text-[var(--muted)] mb-6 max-w-md mx-auto">
            Prepare a project first, then you can test “what if we changed X?” here.
          </p>
          <Link to="/projects" className="btn-primary">
            Go to projects
          </Link>
        </div>
      </div>
    )
  }

  return (
    <div className="page max-w-xl">
      <p className="page-kicker">Scenarios</p>
      <h1 className="page-title">What-if</h1>
      <p className="page-sub">
        Choose a project, then adjust dials and compare before vs after — without opening project
        setup first.
      </p>

      <div className="mt-8 space-y-4">
        <ProjectPicker
          projects={ready}
          value={selected}
          onChange={setSelected}
          id="whatif-entry-project"
          label="Project"
        />
        <button
          type="button"
          className="btn-primary"
          disabled={!selected}
          onClick={() => {
            setLastProjectId(selected)
            navigate(`/whatif/${selected}`)
          }}
        >
          Open scenario workspace
        </button>
        <p className="text-xs text-[var(--muted)]">
          Or review cases first on{' '}
          <Link to="/cases" className="text-teal hover:underline">
            Cases
          </Link>
          .
        </p>
      </div>
    </div>
  )
}
