import React from 'react'
import { BrowserRouter, Routes, Route, Navigate, useParams } from 'react-router-dom'
import { AuthProvider, useAuth } from './context/AuthContext'
import Layout from './components/common/Layout'

import Login from './pages/Login'
import Signup from './pages/Signup'
import Datasets from './pages/Datasets'
import Projects from './pages/Projects'
import ProjectDetail from './pages/ProjectDetail'
import AnalyticsSaaS from './pages/AnalyticsSaaS'
import WhatIf from './pages/WhatIf'

function ExplainabilityRedirect() {
  const { projectId } = useParams()
  return <Navigate to={`/projects/${projectId}`} replace />
}

function Loader() {
  return (
    <div className="min-h-screen flex items-center justify-center bg-paper">
      <div
        className="h-6 w-6 border-2 border-teal border-t-transparent rounded-full animate-spin"
        aria-label="Loading"
      />
    </div>
  )
}

function ProtectedRoute({ children }) {
  const { token, loading } = useAuth()
  if (loading) return <Loader />
  if (!token) return <Navigate to="/login" replace />
  return children
}

function PublicRoute({ children }) {
  const { token, loading } = useAuth()
  if (loading) return <Loader />
  if (token) return <Navigate to="/" replace />
  return children
}

function AppRoutes() {
  return (
    <Routes>
      <Route path="/login" element={<PublicRoute><Login /></PublicRoute>} />
      <Route path="/signup" element={<PublicRoute><Signup /></PublicRoute>} />
      <Route path="/" element={<ProtectedRoute><Layout /></ProtectedRoute>}>
        <Route index element={<Projects />} />
        <Route path="datasets" element={<Datasets />} />
        <Route path="projects" element={<Projects />} />
        <Route path="projects/:id" element={<ProjectDetail />} />
        <Route path="projects/:projectId/explainability" element={<ExplainabilityRedirect />} />
        <Route path="projects/:projectId/whatif" element={<WhatIf />} />
        <Route path="analytics" element={<AnalyticsSaaS />} />
      </Route>
    </Routes>
  )
}

function App() {
  return (
    <BrowserRouter>
      <AuthProvider>
        <AppRoutes />
      </AuthProvider>
    </BrowserRouter>
  )
}

export default App
