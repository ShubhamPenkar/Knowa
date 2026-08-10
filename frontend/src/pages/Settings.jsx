import { useEffect, useMemo, useState } from 'react'
import { Link } from 'react-router-dom'
import { useAuth } from '../context/AuthContext'
import { getLastProjectId, setLastProjectId, clearLastProjectId } from '../lib/lastProject'
import { validateRequired } from '../lib/authValidation'

const INDUSTRIES = [
  { value: 'saas', label: 'SaaS / Software' },
  { value: 'ecommerce', label: 'E-commerce / Retail' },
  { value: 'finance', label: 'Finance / Banking' },
  { value: 'healthcare', label: 'Healthcare' },
  { value: 'other', label: 'Other' },
]

function roleLabel(role) {
  if (role === 'owner') return 'Owner'
  if (role === 'admin') return 'Admin'
  if (role === 'viewer') return 'Viewer'
  return 'Member'
}

export default function Settings() {
  const { user, organization, token, logout, updateProfile, updateOrganization } = useAuth()
  const canEditWorkspace = user?.role === 'owner' || user?.role === 'admin'

  const [name, setName] = useState(user?.name || '')
  const [profileError, setProfileError] = useState('')
  const [profileStatus, setProfileStatus] = useState('')
  const [profileSaving, setProfileSaving] = useState(false)

  const [orgName, setOrgName] = useState(organization?.name || '')
  const [industry, setIndustry] = useState(organization?.industry || 'saas')
  const [orgError, setOrgError] = useState('')
  const [orgStatus, setOrgStatus] = useState('')
  const [orgSaving, setOrgSaving] = useState(false)

  const [projects, setProjects] = useState([])
  const [defaultProjectId, setDefaultProjectId] = useState(getLastProjectId())
  const [defaultStatus, setDefaultStatus] = useState('')

  const [members, setMembers] = useState([])
  const [membersError, setMembersError] = useState('')
  const [membersLoading, setMembersLoading] = useState(true)

  useEffect(() => {
    setName(user?.name || '')
  }, [user?.name])

  useEffect(() => {
    setOrgName(organization?.name || '')
    setIndustry(organization?.industry || 'saas')
  }, [organization?.name, organization?.industry])

  useEffect(() => {
    if (!token) return
    let cancelled = false
    ;(async () => {
      try {
        const [projRes, memRes] = await Promise.all([
          fetch('/api/projects', { headers: { Authorization: `Bearer ${token}` } }),
          fetch('/api/auth/members', { headers: { Authorization: `Bearer ${token}` } }),
        ])
        if (cancelled) return
        if (projRes.ok) {
          const data = await projRes.json()
          setProjects(Array.isArray(data) ? data : [])
        }
        if (memRes.ok) {
          const data = await memRes.json()
          setMembers(data.members || [])
          setMembersError('')
        } else {
          setMembersError('Could not load team members')
        }
      } catch {
        if (!cancelled) setMembersError('Could not load team members')
      } finally {
        if (!cancelled) setMembersLoading(false)
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

  const saveProfile = async (e) => {
    e.preventDefault()
    setProfileError('')
    setProfileStatus('')
    const err = validateRequired(name, 'your name')
    if (err) {
      setProfileError(err)
      return
    }
    setProfileSaving(true)
    try {
      await updateProfile({ name: name.trim() })
      setProfileStatus('Saved')
    } catch (err) {
      setProfileError(err.message || 'Could not save profile')
    }
    setProfileSaving(false)
  }

  const saveWorkspace = async (e) => {
    e.preventDefault()
    if (!canEditWorkspace) return
    setOrgError('')
    setOrgStatus('')
    const err = validateRequired(orgName, 'workspace name')
    if (err) {
      setOrgError(err)
      return
    }
    setOrgSaving(true)
    try {
      await updateOrganization({
        name: orgName.trim(),
        industry: industry || 'other',
      })
      setOrgStatus('Saved')
    } catch (err) {
      setOrgError(err.message || 'Could not save workspace')
    }
    setOrgSaving(false)
  }

  const saveDefaultProject = () => {
    if (defaultProjectId) setLastProjectId(defaultProjectId)
    else clearLastProjectId()
    setDefaultStatus('Saved — Cases and What-if will prefer this project')
  }

  return (
    <div className="page">
      <div className="page-header">
        <div>
          <p className="page-kicker">Account</p>
          <h1 className="page-title">Settings</h1>
          <p className="page-sub">
            Tune your profile, workspace, and defaults — without leaving the decision loop.
          </p>
        </div>
      </div>

      <div className="space-y-8 max-w-2xl">
        <section className="border border-mist" aria-labelledby="profile-heading">
          <div className="px-5 py-4 border-b border-mist">
            <h2 id="profile-heading" className="font-display text-lg font-semibold text-ink">
              Profile
            </h2>
            <p className="text-sm text-[var(--muted)] mt-1">How you appear in follow-ups and the team list.</p>
          </div>
          <form onSubmit={saveProfile} className="px-5 py-5 space-y-4" noValidate>
            <div>
              <label htmlFor="settings-email" className="block text-sm font-medium text-ink mb-1">
                Email
              </label>
              <input
                id="settings-email"
                type="email"
                value={user?.email || ''}
                className="input opacity-80"
                disabled
                readOnly
              />
              <p className="mt-1 text-xs text-[var(--muted)]">Email can&apos;t be changed here yet.</p>
            </div>
            <div>
              <label htmlFor="settings-name" className="block text-sm font-medium text-ink mb-1">
                Display name
              </label>
              <input
                id="settings-name"
                type="text"
                value={name}
                onChange={(e) => {
                  setName(e.target.value)
                  if (profileError) setProfileError('')
                  if (profileStatus) setProfileStatus('')
                }}
                className={`input ${profileError ? 'border-coral/50' : ''}`}
                disabled={profileSaving}
                autoComplete="name"
              />
              {profileError && (
                <p className="mt-1 text-xs text-coral" role="alert">
                  {profileError}
                </p>
              )}
            </div>
            <div className="flex flex-wrap items-center gap-3">
              <button type="submit" className="btn-primary" disabled={profileSaving}>
                {profileSaving ? 'Saving…' : 'Save profile'}
              </button>
              {profileStatus && <span className="text-sm text-teal">{profileStatus}</span>}
              {user?.role && (
                <span className="text-xs text-[var(--muted)] ml-auto">
                  Role · {roleLabel(user.role)}
                </span>
              )}
            </div>
          </form>
        </section>

        <section className="border border-mist" aria-labelledby="workspace-heading">
          <div className="px-5 py-4 border-b border-mist">
            <h2 id="workspace-heading" className="font-display text-lg font-semibold text-ink">
              Workspace
            </h2>
            <p className="text-sm text-[var(--muted)] mt-1">
              {organization?.slug ? `Slug · ${organization.slug}` : 'Organization details'}
            </p>
          </div>
          <form onSubmit={saveWorkspace} className="px-5 py-5 space-y-4" noValidate>
            <div>
              <label htmlFor="settings-org-name" className="block text-sm font-medium text-ink mb-1">
                Workspace name
              </label>
              <input
                id="settings-org-name"
                type="text"
                value={orgName}
                onChange={(e) => {
                  setOrgName(e.target.value)
                  if (orgError) setOrgError('')
                  if (orgStatus) setOrgStatus('')
                }}
                className={`input ${orgError ? 'border-coral/50' : ''}`}
                disabled={!canEditWorkspace || orgSaving}
              />
            </div>
            <div>
              <label htmlFor="settings-industry" className="block text-sm font-medium text-ink mb-1">
                Industry
              </label>
              <select
                id="settings-industry"
                value={industry || 'other'}
                onChange={(e) => {
                  setIndustry(e.target.value)
                  if (orgStatus) setOrgStatus('')
                }}
                className="input"
                disabled={!canEditWorkspace || orgSaving}
              >
                {INDUSTRIES.map((opt) => (
                  <option key={opt.value} value={opt.value}>
                    {opt.label}
                  </option>
                ))}
              </select>
            </div>
            {!canEditWorkspace && (
              <p className="text-sm text-[var(--muted)]">
                Only owners and admins can change workspace details.
              </p>
            )}
            {orgError && (
              <p className="text-xs text-coral" role="alert">
                {orgError}
              </p>
            )}
            {canEditWorkspace && (
              <div className="flex flex-wrap items-center gap-3">
                <button type="submit" className="btn-primary" disabled={orgSaving}>
                  {orgSaving ? 'Saving…' : 'Save workspace'}
                </button>
                {orgStatus && <span className="text-sm text-teal">{orgStatus}</span>}
              </div>
            )}
          </form>
        </section>

        <section className="border border-mist" aria-labelledby="defaults-heading">
          <div className="px-5 py-4 border-b border-mist">
            <h2 id="defaults-heading" className="font-display text-lg font-semibold text-ink">
              Defaults
            </h2>
            <p className="text-sm text-[var(--muted)] mt-1">
              Preferred project for Cases and What-if when you open those hubs.
            </p>
          </div>
          <div className="px-5 py-5 space-y-4">
            {readyProjects.length === 0 ? (
              <p className="text-sm text-[var(--muted)]">
                No ready projects yet.{' '}
                <Link to="/projects" className="text-teal hover:underline">
                  Set up a project
                </Link>
              </p>
            ) : (
              <>
                <div>
                  <label
                    htmlFor="settings-default-project"
                    className="block text-sm font-medium text-ink mb-1"
                  >
                    Default project
                  </label>
                  <select
                    id="settings-default-project"
                    value={defaultProjectId || ''}
                    onChange={(e) => {
                      setDefaultProjectId(e.target.value)
                      setDefaultStatus('')
                    }}
                    className="input"
                  >
                    <option value="">No preference</option>
                    {readyProjects.map((p) => (
                      <option key={p.id} value={p.id}>
                        {p.name}
                      </option>
                    ))}
                  </select>
                </div>
                <div className="flex flex-wrap items-center gap-3">
                  <button type="button" className="btn-primary" onClick={saveDefaultProject}>
                    Save default
                  </button>
                  {defaultStatus && <span className="text-sm text-teal">{defaultStatus}</span>}
                </div>
              </>
            )}
          </div>
        </section>

        <section className="border border-mist" aria-labelledby="team-heading">
          <div className="px-5 py-4 border-b border-mist">
            <h2 id="team-heading" className="font-display text-lg font-semibold text-ink">
              Team
            </h2>
            <p className="text-sm text-[var(--muted)] mt-1">
              People in this workspace. Invites come later — assignees already use this list.
            </p>
          </div>
          <div className="px-5 py-5">
            {membersLoading ? (
              <div className="space-y-2" aria-busy="true">
                <div className="skeleton h-10 w-full" />
                <div className="skeleton h-10 w-full" />
              </div>
            ) : membersError ? (
              <p className="text-sm text-coral">{membersError}</p>
            ) : members.length === 0 ? (
              <p className="text-sm text-[var(--muted)]">No members found.</p>
            ) : (
              <ul className="divide-y divide-mist border border-mist">
                {members.map((m) => (
                  <li
                    key={m.id}
                    className="px-4 py-3 flex flex-wrap items-center justify-between gap-2"
                  >
                    <div className="min-w-0">
                      <p className="text-sm text-ink font-medium truncate">
                        {m.name}
                        {m.id === user?.id ? (
                          <span className="text-[var(--muted)] font-normal"> · you</span>
                        ) : null}
                      </p>
                      <p className="text-xs text-[var(--muted)] truncate">{m.email}</p>
                    </div>
                    <span className="badge bg-mist/60 text-ink shrink-0">{roleLabel(m.role)}</span>
                  </li>
                ))}
              </ul>
            )}
          </div>
        </section>

        <section className="border border-mist" aria-labelledby="session-heading">
          <div className="px-5 py-4 border-b border-mist">
            <h2 id="session-heading" className="font-display text-lg font-semibold text-ink">
              Session
            </h2>
          </div>
          <div className="px-5 py-5 flex flex-wrap items-center gap-3">
            <button type="button" className="btn-secondary" onClick={logout}>
              Sign out
            </button>
            <p className="text-sm text-[var(--muted)]">Ends this browser session.</p>
          </div>
        </section>
      </div>
    </div>
  )
}
