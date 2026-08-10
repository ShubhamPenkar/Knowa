import { useState, useEffect, useMemo, useRef } from 'react'
import { Link, useSearchParams } from 'react-router-dom'
import { useAuth } from '../context/AuthContext'
import { PredictionPanel } from '../components/PredictionPanel'
import ProjectPicker from '../components/common/ProjectPicker'
import OrgHealthStrip from '../components/common/OrgHealthStrip'
import Spinner from '../components/common/Spinner'
import { getLastProjectId, setLastProjectId } from '../lib/lastProject'
import { softReasonLabel } from '../lib/trustAssessment'

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
 * Top-level Cases hub — review + Don't-act (soft/low-trust) queue.
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
  const [softScan, setSoftScan] = useState([])
  const [softScanLoading, setSoftScanLoading] = useState(false)
  const [softScanError, setSoftScanError] = useState('')
  const [storedSoft, setStoredSoft] = useState([])
  const predictSeq = useRef(0)

  const filterMode = searchParams.get('filter') === 'dont-act' ? 'dont-act' : 'all'

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
      setSoftScan([])
      setStoredSoft([])
      return
    }
    setLastProjectId(projectId)
    predictSeq.current += 1
    setSelectedRowIdx(null)
    setPrediction(null)
    setPredictError('')
    setKnownOutcome(null)
    setSoftScan([])
    setRowsLoading(true)
    ;(async () => {
      try {
        const [pRes, tRes, predRes] = await Promise.all([
          fetch(`/api/projects/${projectId}`, {
            headers: { Authorization: `Bearer ${token}` },
          }),
          fetch(`/api/projects/${projectId}/test-data?limit=30`, {
            headers: { Authorization: `Bearer ${token}` },
          }),
          fetch(`/api/projects/${projectId}/predictions?soft_only=true&limit=100`, {
            headers: { Authorization: `Bearer ${token}` },
          }),
        ])
        if (pRes.ok) setProject(await pRes.json())
        else setProject(null)
        if (tRes.ok) setTestData(await tRes.json())
        else setTestData(null)
        if (predRes.ok) {
          const preds = await predRes.json()
          setStoredSoft(Array.isArray(preds) ? preds : [])
        } else {
          setStoredSoft([])
        }
      } catch {
        setProject(null)
        setTestData(null)
        setStoredSoft([])
      }
      setRowsLoading(false)
    })()
  }, [token, projectId])

  // Prefetch Don't-act scan so the tab badge stays accurate
  useEffect(() => {
    if (!token || !projectId || !project || !testData?.rows?.length) {
      setSoftScan([])
      setSoftScanError('')
      setSoftScanLoading(false)
      return
    }
    let cancelled = false
    ;(async () => {
      setSoftScanLoading(true)
      setSoftScanError('')
      try {
        const rows = (testData.rows || []).slice(0, 20)
        const res = await fetch(`/api/projects/${projectId}/predict/batch`, {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json',
            Authorization: `Bearer ${token}`,
          },
          body: JSON.stringify({
            rows,
            max_rows: 20,
          }),
        })
        const data = await res.json().catch(() => ({}))
        if (cancelled) return
        if (res.ok) {
          const soft = (data.results || []).filter((r) => r.ok && (r.soft_case || r.low_confidence))
          setSoftScan(soft)
          setSoftScanError('')
        } else {
          setSoftScan([])
          setSoftScanError(
            typeof data.detail === 'string'
              ? data.detail
              : 'Could not scan the Don\u2019t-act queue'
          )
        }
      } catch (err) {
        if (!cancelled) {
          setSoftScan([])
          setSoftScanError(err?.message || 'Could not scan the Don\u2019t-act queue')
        }
      }
      if (!cancelled) setSoftScanLoading(false)
    })()
    return () => {
      cancelled = true
    }
  }, [token, projectId, project, testData])

  // Deep-link hydrate: /cases?project=&prediction=
  useEffect(() => {
    const predictionId = searchParams.get('prediction')
    if (!token || !projectId || !predictionId) return
    let cancelled = false
    const seq = ++predictSeq.current
    ;(async () => {
      setPredicting(true)
      setPredictError('')
      setPrediction(null)
      try {
        const res = await fetch(`/api/projects/${projectId}/predictions/${predictionId}`, {
          headers: { Authorization: `Bearer ${token}` },
        })
        const data = await res.json().catch(() => ({}))
        if (cancelled || seq !== predictSeq.current) return
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
        if (!cancelled && seq === predictSeq.current) {
          setPredictError(err?.message || 'Could not open that case')
        }
      }
      if (!cancelled && seq === predictSeq.current) setPredicting(false)
    })()
    return () => {
      cancelled = true
    }
  }, [token, projectId, searchParams])

  const setFilterMode = (mode) => {
    const next = new URLSearchParams(searchParams)
    if (mode === 'dont-act') next.set('filter', 'dont-act')
    else next.delete('filter')
    setSearchParams(next, { replace: true })
    setPrediction(null)
    setSelectedRowIdx(null)
  }

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

  // Live sample softs that aren't already in the logged queue (entity match)
  const loggedEntityIds = useMemo(() => {
    const ids = new Set()
    for (const p of storedSoft) {
      if (p.entity_id) ids.add(String(p.entity_id))
    }
    return ids
  }, [storedSoft])

  const sampleSoftOnly = useMemo(
    () =>
      softScan.filter((item) => {
        const eid = item.entity_id != null ? String(item.entity_id) : ''
        return !eid || !loggedEntityIds.has(eid)
      }),
    [softScan, loggedEntityIds]
  )

  const loggedDontActCount = storedSoft.length
  const sampleDontActCount = sampleSoftOnly.length

  const handlePredictRow = async (row, idx) => {
    if (!project) return
    const seq = ++predictSeq.current
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
      if (seq !== predictSeq.current) return
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
      if (seq !== predictSeq.current) return
      setPredictError('Network error: ' + err.message)
    }
    if (seq === predictSeq.current) setPredicting(false)
  }

  const openSoftScanCase = async (item) => {
    const idx = item.index
    const row = rows[idx]
    if (!row) {
      setPredictError('That sample row is no longer loaded — refresh Cases and try again.')
      return
    }
    await handlePredictRow(row, idx)
  }

  const openStoredSoft = async (pred) => {
    const seq = ++predictSeq.current
    setPredicting(true)
    setPredictError('')
    setSelectedRowIdx(null)
    setKnownOutcome(null)
    setPrediction(null)
    try {
      const res = await fetch(`/api/projects/${projectId}/predictions/${pred.id}`, {
        headers: { Authorization: `Bearer ${token}` },
      })
      const data = await res.json().catch(() => ({}))
      if (seq !== predictSeq.current) return
      if (!res.ok) {
        setPredictError(typeof data.detail === 'string' ? data.detail : 'Could not open case')
      } else {
        setPrediction(data)
      }
    } catch (err) {
      if (seq !== predictSeq.current) return
      setPredictError(err?.message || 'Could not open case')
    }
    if (seq === predictSeq.current) setPredicting(false)
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
            Review cases that need attention — use Don't act when the score isn't firm enough to spend on.
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

      <OrgHealthStrip projectId={projectId} />

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
          <div className="mb-6 flex flex-wrap items-end justify-between gap-4">
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

          <div className="flex flex-wrap gap-2 mb-6" role="tablist" aria-label="Case queue">
            {[
              { id: 'all', label: 'All sample cases' },
              {
                id: 'dont-act',
                label: `Don't act${
                  softScanLoading && !loggedDontActCount
                    ? '…'
                    : loggedDontActCount
                      ? ` (${loggedDontActCount})`
                      : softScanError
                        ? ' (!)'
                        : ''
                }`,
              },
            ].map((tab) => (
              <button
                key={tab.id}
                type="button"
                role="tab"
                aria-selected={filterMode === tab.id}
                onClick={() => setFilterMode(tab.id)}
                className={`px-3 py-1.5 text-xs border rounded-control ${
                  filterMode === tab.id
                    ? 'border-teal bg-teal-soft/30 text-ink'
                    : 'border-mist text-[var(--muted)] hover:text-ink'
                }`}
              >
                {tab.label}
              </button>
            ))}
          </div>

          {rowsLoading ? (
            <div className="border border-mist px-5 py-10 space-y-4" aria-busy="true">
              <div className="flex items-center gap-3 text-sm text-[var(--muted)]">
                <Spinner className="h-4 w-4" /> Loading cases…
              </div>
              <div className="space-y-2">
                {[0, 1, 2].map((i) => (
                  <div key={i} className="skeleton h-10 w-full" />
                ))}
              </div>
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
                    <div className="case-brief-stage px-5 py-6 text-sm text-[var(--muted)] flex items-center gap-3">
                      <Spinner className="h-4 w-4" /> Opening brief…
                    </div>
                  )}
                  {!predicting && !prediction && (
                    <div className="case-brief-stage px-5 py-6 md:px-6">
                      <p className="page-kicker">Brief</p>
                      <h3 className="font-display text-xl font-semibold text-ink mt-1 tracking-tight">
                        {filterMode === 'dont-act'
                          ? "Pick a Don't-act case"
                          : 'Select someone to begin'}
                      </h3>
                      <p className="text-sm text-[var(--muted)] mt-2 leading-relaxed max-w-sm">
                        {filterMode === 'dont-act'
                          ? "Scores here aren't firm enough for big spends — prefer light check-ins and confirm with what-if first."
                          : `You'll see how likely ${outcomeLabel.toLowerCase()} is, what's driving it, and what to do next.`}
                      </p>
                    </div>
                  )}
                  {!predicting && prediction && (
                    <div
                      key={
                        prediction.prediction_id ||
                        prediction.id ||
                        `row-${selectedRowIdx ?? 'case'}`
                      }
                      className="case-open space-y-3"
                    >
                      {(prediction.low_confidence ||
                        prediction.abstention_reason ||
                        filterMode === 'dont-act') && (
                        <div className="border border-coral/30 bg-coral-soft/40 px-3 py-2 text-xs text-ink rounded-control">
                          Don&apos;t act yet — review carefully before spending.
                        </div>
                      )}
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
                    </div>
                  )}
                </div>
              </div>

              <div
                className={`lg:col-span-3 min-w-0 ${prediction || predicting ? 'order-2' : 'order-1'} lg:order-1`}
              >
                {filterMode === 'dont-act' ? (
                  <>
                    <h2 className="font-display text-lg font-semibold text-ink mb-1">
                      Don&apos;t act yet
                    </h2>
                    <p className="text-sm text-[var(--muted)] mb-4">
                      <span className="text-ink tabular-nums">{loggedDontActCount}</span> logged
                      {sampleDontActCount
                        ? ` · ${sampleDontActCount} also soft in this sample (not logged yet)`
                        : ''}
                      . Prefer light check-ins when certainty is soft.
                    </p>
                    {softScanError && loggedDontActCount === 0 && sampleDontActCount === 0 && !softScanLoading ? (
                      <div className="empty-state py-10">
                        <h3 className="font-display text-lg font-semibold text-ink mb-2">
                          Couldn&apos;t scan Don&apos;t act
                        </h3>
                        <p className="text-sm text-[var(--muted)] mb-5 max-w-md mx-auto leading-relaxed">
                          {softScanError}
                        </p>
                        <button
                          type="button"
                          onClick={() => setFilterMode('all')}
                          className="btn-secondary text-sm"
                        >
                          Browse all sample cases
                        </button>
                      </div>
                    ) : !softScanLoading &&
                      loggedDontActCount === 0 &&
                      sampleDontActCount === 0 ? (
                      <div className="empty-state py-10">
                        <h3 className="font-display text-lg font-semibold text-ink mb-2">
                          Queue is clear
                        </h3>
                        <p className="text-sm text-[var(--muted)] mb-5 max-w-md mx-auto leading-relaxed">
                          Nothing logged as Don&apos;t act for this project, and this sample looks
                          firm. Open more cases if you want low-trust flags to accumulate.
                        </p>
                        <button
                          type="button"
                          onClick={() => setFilterMode('all')}
                          className="btn-secondary text-sm"
                        >
                          Browse all sample cases
                        </button>
                      </div>
                    ) : (
                      <div className="space-y-6">
                        {softScanError && (
                          <p className="text-xs text-coral border border-coral/30 bg-coral-soft/30 px-3 py-2">
                            Live sample scan issue: {softScanError}. Logged cases below still work.
                          </p>
                        )}

                        <div>
                          <h3 className="text-[11px] font-semibold uppercase tracking-[0.14em] text-teal mb-2">
                            Logged ({loggedDontActCount})
                          </h3>
                          <p className="text-xs text-[var(--muted)] mb-3">
                            Matches &quot;Logged don&apos;t act&quot; for this project — saved when you
                            opened a soft brief.
                          </p>
                          {loggedDontActCount === 0 ? (
                            <p className="text-sm text-[var(--muted)] border border-mist px-4 py-5">
                              No logged Don&apos;t-act cases yet for this project.
                            </p>
                          ) : (
                            <ul className="border border-mist divide-y divide-mist">
                              {storedSoft.slice(0, 40).map((pred, i) => (
                                <li
                                  key={pred.id}
                                  className="list-row-in"
                                  style={{ animationDelay: `${60 + i * 45}ms` }}
                                >
                                  <button
                                    type="button"
                                    onClick={() => openStoredSoft(pred)}
                                    className="w-full text-left px-4 py-3.5 hover:bg-mist/30 transition-colors"
                                  >
                                    <div className="flex flex-wrap items-center gap-2">
                                      <span className="badge bg-coral-soft border border-coral/30 text-ink">
                                        Logged
                                      </span>
                                      <span className="text-sm text-ink">
                                        {pred.entity_id || pred.id.slice(0, 8)}
                                      </span>
                                    </div>
                                    <p className="text-xs text-[var(--muted)] mt-1">
                                      {pred.probability != null
                                        ? `About ${Math.round(Number(pred.probability) * 100)}%`
                                        : 'Stored case'}
                                      {pred.created_at
                                        ? ` · ${new Date(pred.created_at).toLocaleDateString()}`
                                        : ''}
                                      {' · Model is less sure'}
                                    </p>
                                  </button>
                                </li>
                              ))}
                            </ul>
                          )}
                        </div>

                        <div>
                          <h3 className="text-[11px] font-semibold uppercase tracking-[0.14em] text-teal mb-2">
                            Also soft in this sample
                            {softScanLoading ? '…' : ` (${sampleDontActCount})`}
                          </h3>
                          <p className="text-xs text-[var(--muted)] mb-3">
                            Live scan — not counted in Logged until you open the brief (that
                            saves it).
                          </p>
                          {softScanLoading ? (
                            <div className="border border-mist px-4 py-5 flex items-center gap-3 text-sm text-[var(--muted)]">
                              <Spinner className="h-4 w-4" /> Scanning sample…
                            </div>
                          ) : sampleDontActCount === 0 ? (
                            <p className="text-sm text-[var(--muted)] border border-mist px-4 py-5">
                              No extra soft cases in the current sample beyond what&apos;s already
                              logged.
                            </p>
                          ) : (
                            <ul className="border border-mist divide-y divide-mist">
                              {sampleSoftOnly.map((item, i) => {
                                const pct =
                                  item.probability != null
                                    ? Math.round(Number(item.probability) * 100)
                                    : null
                                const reason = softReasonLabel(item.soft_reason)
                                return (
                                  <li
                                    key={`scan-${item.index}`}
                                    className="list-row-in"
                                    style={{ animationDelay: `${60 + i * 45}ms` }}
                                  >
                                    <button
                                      type="button"
                                      onClick={() => openSoftScanCase(item)}
                                      className="w-full text-left px-4 py-3.5 hover:bg-mist/30 transition-colors"
                                    >
                                      <div className="flex flex-wrap items-center gap-2">
                                        <span className="badge bg-mist text-ink border border-mist">
                                          Sample
                                        </span>
                                        <span className="text-sm text-ink">
                                          Case #{item.index + 1}
                                          {item.entity_id ? ` · ${item.entity_id}` : ''}
                                        </span>
                                      </div>
                                      <p className="text-xs text-[var(--muted)] mt-1">
                                        {pct != null
                                          ? `About ${pct}% likelihood`
                                          : 'Elevated uncertainty'}
                                        {reason ? ` · ${reason}` : ''}
                                      </p>
                                    </button>
                                  </li>
                                )
                              })}
                            </ul>
                          )}
                        </div>
                      </div>
                    )}
                  </>
                ) : (
                  <>
                    <h2 className="font-display text-lg font-semibold text-ink mb-1">
                      Who needs attention?
                    </h2>
                    <p className="text-sm text-[var(--muted)] mb-4">
                      Click a row to open their brief. Save a follow-up when you decide to act.
                    </p>
                    {rows.length === 0 ? (
                      <div className="empty-state py-10">
                        <h3 className="font-display text-lg font-semibold text-ink mb-2">
                          No sample cases yet
                        </h3>
                        <p className="text-sm text-[var(--muted)] mb-5 max-w-md mx-auto leading-relaxed">
                          This project doesn&apos;t have sample rows to review. Refresh data from
                          project settings.
                        </p>
                        <Link to={`/projects/${projectId}`} className="btn-primary text-sm">
                          Open project settings
                        </Link>
                      </div>
                    ) : (
                      <div className="border border-mist overflow-x-auto bg-paper/40">
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
                                  className={`cursor-pointer transition-colors list-row-fade hover:bg-mist/40 ${
                                    selected ? 'bg-teal-soft/30 ring-1 ring-inset ring-teal/30' : ''
                                  }`}
                                  style={{ animationDelay: `${40 + idx * 28}ms` }}
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
                  </>
                )}
              </div>
            </section>
          )}
        </>
      )}
    </div>
  )
}
