import { useState, useEffect, useMemo } from 'react'
import { Link, useSearchParams } from 'react-router-dom'
import { useAuth } from '../context/AuthContext'
import { PredictionPanel } from '../components/PredictionPanel'
import ProjectPicker from '../components/common/ProjectPicker'
import Spinner from '../components/common/Spinner'
import { getLastProjectId, setLastProjectId } from '../lib/lastProject'

function resolveEntityId(row, idx) {
  if (!row) return `row-${idx}`
  const preferred = [
    'customerID',
    'CustomerID',
    'customer_id',
    'customerId',
    'entity_id',
    'EntityId',
    'account_id',
    'AccountID',
    'user_id',
    'UserID',
    'id',
    'ID',
  ]
  for (const k of preferred) {
    if (row[k] != null && String(row[k]).trim() !== '') return String(row[k])
  }
  return `row-${idx}`
}

function formatApiError(detail) {
  if (detail == null) return 'Request failed'
  if (typeof detail === 'string') return detail
  if (Array.isArray(detail)) {
    return detail.map((d) => d.msg || JSON.stringify(d)).join('; ')
  }
  if (typeof detail === 'object' && detail.message) return detail.message
  return String(detail)
}

/**
 * Top-level Cases hub — review who needs attention without opening project setup.
 */
export default function Cases() {
  const { token } = useAuth()
  const [searchParams, setSearchParams] = useSearchParams()
  const [projects, setProjects] = useState([])
  const [projectId, setProjectId] = useState('')
  const [project, setProject] = useState(null)
  const [loading, setLoading] = useState(true)
  const [rowsLoading, setRowsLoading] = useState(false)
  const [testData, setTestData] = useState(null)
  const [selectedRowIdx, setSelectedRowIdx] = useState(null)
  const [prediction, setPrediction] = useState(null)
  const [predicting, setPredicting] = useState(false)
  const [predictError, setPredictError] = useState('')
  const [knownOutcome, setKnownOutcome] = useState(null)

  const readyProjects = useMemo(
    () => projects.filter((p) => p.status === 'ready' || p.status === 'trained'),
    [projects]
  )

  useEffect(() => {
    if (!token) return
    ;(async () => {
      try {
        const res = await fetch('/api/projects', {
          headers: { Authorization: `Bearer ${token}` },
        })
        if (res.ok) {
          const data = await res.json()
          setProjects(data)
          const ready = data.filter((p) => p.status === 'ready' || p.status === 'trained')
          const fromQuery = searchParams.get('project')
          const last = getLastProjectId()
          const pick =
            ready.find((p) => p.id === fromQuery)?.id ||
            ready.find((p) => p.id === last)?.id ||
            ready[0]?.id ||
            ''
          setProjectId(pick)
        }
      } catch (err) {
        console.error(err)
      }
      setLoading(false)
    })()
  }, [token])

  useEffect(() => {
    if (!token || !projectId) {
      setProject(null)
      setTestData(null)
      setPrediction(null)
      return
    }
    setLastProjectId(projectId)
    setSelectedRowIdx(null)
    setPrediction(null)
    setPredictError('')
    setKnownOutcome(null)
    setRowsLoading(true)
    ;(async () => {
      try {
        const [pRes, tRes] = await Promise.all([
          fetch(`/api/projects/${projectId}`, {
            headers: { Authorization: `Bearer ${token}` },
          }),
          fetch(`/api/projects/${projectId}/test-data?limit=30`, {
            headers: { Authorization: `Bearer ${token}` },
          }),
        ])
        if (pRes.ok) setProject(await pRes.json())
        else setProject(null)
        if (tRes.ok) setTestData(await tRes.json())
        else setTestData(null)
      } catch {
        setProject(null)
        setTestData(null)
      }
      setRowsLoading(false)
    })()
  }, [token, projectId])

  // Deep-link hydrate: /cases?project=&prediction=
  useEffect(() => {
    const predictionId = searchParams.get('prediction')
    if (!token || !projectId || !predictionId) return
    let cancelled = false
    ;(async () => {
      setPredicting(true)
      setPredictError('')
      try {
        const res = await fetch(`/api/projects/${projectId}/predictions/${predictionId}`, {
          headers: { Authorization: `Bearer ${token}` },
        })
        const data = await res.json().catch(() => ({}))
        if (cancelled) return
        if (!res.ok) {
          setPredictError(
            typeof data.detail === 'string' ? data.detail : 'Could not open that case'
          )
        } else {
          setPrediction(data)
          setSelectedRowIdx(null)
          setKnownOutcome(null)
        }
      } catch (err) {
        if (!cancelled) setPredictError(err?.message || 'Could not open that case')
      }
      if (!cancelled) setPredicting(false)
    })()
    return () => {
      cancelled = true
    }
  }, [token, projectId, searchParams])

  const handleProjectChange = (id) => {
    setProjectId(id)
    const next = new URLSearchParams(searchParams)
    if (id) next.set('project', id)
    else next.delete('project')
    next.delete('prediction')
    setSearchParams(next, { replace: true })
  }

  const rows = testData?.rows || []
  const columnSource = (rows[0] && Object.keys(rows[0])) || []
  const featurePreviewCols = columnSource
    .filter((col) => col !== project?.target_column)
    .slice(0, 5)
  const mobileFeatureCols = featurePreviewCols.slice(0, 1)

  const outcomeLabel = (project?.target_description || project?.target_column || 'outcome')
    .replace(/[_-]+/g, ' ')
    .trim()

  const handlePredictRow = async (row, idx) => {
    if (!project) return
    setSelectedRowIdx(idx)
    setPredicting(true)
    setPredictError('')
    setPrediction(null)
    const actualKnown =
      project.target_column != null
        ? String(row[project.target_column]) === String(project.target_positive_label)
        : null
    setKnownOutcome(actualKnown)

    const features = {}
    ;(project.feature_columns || []).forEach((col) => {
      features[col] = row[col]
    })

    try {
      const res = await fetch(`/api/projects/${projectId}/predict`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          Authorization: `Bearer ${token}`,
        },
        body: JSON.stringify({
          features,
          entity_id: resolveEntityId(row, idx),
          include_explanations: true,
          include_recommendations: true,
        }),
      })
      const data = await res.json()
      if (res.ok) {
        setPrediction(data)
        requestAnimationFrame(() => {
          document
            .getElementById('cases-brief')
            ?.scrollIntoView({ behavior: 'smooth', block: 'nearest' })
        })
      } else {
        setPredictError(formatApiError(data.detail) || 'Prediction failed')
      }
    } catch (err) {
      setPredictError('Network error: ' + err.message)
    }
    setPredicting(false)
  }

  if (loading) {
    return (
      <div className="page flex justify-center items-center min-h-[40vh]">
        <Spinner />
      </div>
    )
  }

  return (
    <div className="page">
      <div className="page-header">
        <div>
          <p className="page-kicker">Review</p>
          <h1 className="page-title">Cases</h1>
          <p className="page-sub max-w-xl">
            Open people or accounts that need attention, see why, and save a follow-up — without
            digging through project setup.
          </p>
        </div>
        <div className="flex flex-wrap gap-2">
          {projectId && (
            <>
              <Link to={`/whatif/${projectId}`} className="btn-secondary text-sm">
                What-if
              </Link>
              <Link to={`/projects/${projectId}`} className="btn-ghost text-sm">
                Project settings
              </Link>
            </>
          )}
        </div>
      </div>

      {readyProjects.length === 0 ? (
        <div className="empty-state">
          <h3 className="font-display text-xl font-semibold text-ink mb-2">Nothing ready yet</h3>
          <p className="text-sm text-[var(--muted)] mb-6 max-w-md mx-auto">
            Prepare a project first — then cases to review show up here.
          </p>
          <Link to="/projects" className="btn-primary">
            Go to projects
          </Link>
        </div>
      ) : (
        <>
          <div className="mb-8 flex flex-wrap items-end justify-between gap-4">
            <ProjectPicker
              projects={readyProjects}
              value={projectId}
              onChange={handleProjectChange}
              id="cases-project"
              label="Looking at"
            />
            {project && (
              <p className="text-sm text-[var(--muted)] pb-1">
                Watching for <span className="text-ink capitalize">{outcomeLabel}</span>
              </p>
            )}
          </div>

          {rowsLoading ? (
            <div className="flex items-center gap-3 text-sm text-[var(--muted)] py-10">
              <Spinner className="h-4 w-4" /> Loading cases…
            </div>
          ) : (
            <section className="grid lg:grid-cols-5 gap-8">
              <div
                id="cases-brief"
                className={`lg:col-span-2 ${prediction || predicting ? 'order-1' : 'order-2'} lg:order-2`}
              >
                <div className="lg:sticky lg:top-6 space-y-3">
                  {predictError && (
                    <div className="text-sm border border-coral/40 bg-coral-soft px-4 py-3 rounded-control">
                      {predictError}
                    </div>
                  )}
                  {predicting && (
                    <div className="surface p-6 text-sm text-[var(--muted)] flex items-center gap-3">
                      <Spinner className="h-4 w-4" /> Opening brief…
                    </div>
                  )}
                  {!predicting && !prediction && (
                    <div className="surface p-6">
                      <p className="page-kicker">Brief</p>
                      <h3 className="font-display text-lg font-semibold text-ink mt-1">
                        Select someone to begin
                      </h3>
                      <p className="text-sm text-[var(--muted)] mt-2 leading-relaxed">
                        You&apos;ll see how likely {outcomeLabel.toLowerCase()} is, what&apos;s
                        driving it, and what to do next.
                      </p>
                    </div>
                  )}
                  {!predicting && prediction && (
                    <PredictionPanel
                      result={prediction}
                      knownOutcome={knownOutcome}
                      outcomeYesLabel="Yes"
                      outcomeNoLabel="No"
                      projectId={projectId}
                      authToken={token}
                      simulateHref={`/whatif/${projectId}${
                        selectedRowIdx != null ? `?row=${selectedRowIdx}` : ''
                      }`}
                      simulateLabel="Explore a what-if"
                    />
                  )}
                </div>
              </div>

              <div
                className={`lg:col-span-3 min-w-0 ${prediction || predicting ? 'order-2' : 'order-1'} lg:order-1`}
              >
                <h2 className="font-display text-lg font-semibold text-ink mb-1">
                  Who needs attention?
                </h2>
                <p className="text-sm text-[var(--muted)] mb-4">
                  Click a row to open their brief. Save a follow-up from the brief when you decide to
                  act.
                </p>
                {rows.length === 0 ? (
                  <div className="border border-mist px-5 py-8 text-sm text-[var(--muted)]">
                    No sample cases for this project yet. Open{' '}
                    <Link to={`/projects/${projectId}`} className="text-teal hover:underline">
                      project settings
                    </Link>{' '}
                    and refresh data.
                  </div>
                ) : (
                  <div className="surface overflow-x-auto">
                    <table className="data-table">
                      <thead>
                        <tr>
                          <th>#</th>
                          <th>In data</th>
                          {mobileFeatureCols.map((col) => (
                            <th key={col} className="truncate max-w-[7rem] md:hidden">
                              {String(col).replace(/[_-]+/g, ' ')}
                            </th>
                          ))}
                          {featurePreviewCols.map((col) => (
                            <th
                              key={`d-${col}`}
                              className="truncate max-w-[6rem] hidden md:table-cell"
                            >
                              {String(col).replace(/[_-]+/g, ' ')}
                            </th>
                          ))}
                        </tr>
                      </thead>
                      <tbody>
                        {rows.slice(0, 15).map((row, idx) => {
                          const actual =
                            String(row[project.target_column]) ===
                            String(project.target_positive_label)
                          const selected = selectedRowIdx === idx
                          return (
                            <tr
                              key={idx}
                              onClick={() => handlePredictRow(row, idx)}
                              onKeyDown={(e) => {
                                if (e.key === 'Enter' || e.key === ' ') {
                                  e.preventDefault()
                                  handlePredictRow(row, idx)
                                }
                              }}
                              tabIndex={0}
                              role="button"
                              aria-pressed={selected}
                              className={`cursor-pointer transition-colors hover:bg-mist/40 ${
                                selected ? 'bg-teal-soft/30 ring-1 ring-inset ring-teal/30' : ''
                              }`}
                            >
                              <td className="text-[var(--muted)]">{idx + 1}</td>
                              <td>
                                <span
                                  className={`badge ${
                                    actual ? 'bg-coral-soft text-ink' : 'bg-mist text-ink'
                                  }`}
                                >
                                  {actual ? 'Yes' : 'No'}
                                </span>
                              </td>
                              {mobileFeatureCols.map((col) => (
                                <td
                                  key={col}
                                  className="truncate max-w-[7rem] text-[var(--muted)] md:hidden"
                                >
                                  {String(row[col] ?? '')}
                                </td>
                              ))}
                              {featurePreviewCols.map((col) => (
                                <td
                                  key={`d-${col}`}
                                  className="truncate max-w-[6rem] text-[var(--muted)] hidden md:table-cell"
                                >
                                  {String(row[col] ?? '')}
                                </td>
                              ))}
                            </tr>
                          )
                        })}
                      </tbody>
                    </table>
                  </div>
                )}
              </div>
            </section>
          )}
        </>
      )}
    </div>
  )
}
