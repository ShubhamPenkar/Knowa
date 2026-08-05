import { useState, createContext, useContext, useEffect } from 'react';

const AuthContext = createContext(null);

async function parseJsonResponse(res, fallbackMessage) {
  const text = await res.text();
  if (!text) {
    if (!res.ok) {
      throw new Error(
        res.status === 500 || res.status === 502 || res.status === 504
          ? 'Backend is not running. Start it with: cd backend && uvicorn app.main:app --reload'
          : fallbackMessage
      );
    }
    return {};
  }
  try {
    return JSON.parse(text);
  } catch {
    throw new Error(
      res.ok
        ? 'Invalid response from server'
        : `Server error (${res.status}). Is the backend running on port 8000?`
    );
  }
}

function formatErrorDetail(detail, fallback) {
  if (!detail) return fallback;
  if (typeof detail === 'string') return detail;
  if (Array.isArray(detail)) {
    return detail.map((e) => e.msg || JSON.stringify(e)).join(', ');
  }
  return fallback;
}

export function AuthProvider({ children }) {
  const [user, setUser] = useState(null);
  const [organization, setOrganization] = useState(null);
  const [token, setToken] = useState(localStorage.getItem('token'));
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    if (token) {
      fetchMe();
    } else {
      setLoading(false);
    }
  }, []);

  const fetchMe = async () => {
    try {
      const res = await fetch('/api/auth/me', {
        headers: { Authorization: `Bearer ${token}` }
      });
      if (res.ok) {
        const data = await parseJsonResponse(res, 'Session check failed');
        setUser(data.user);
        setOrganization(data.organization);
      } else {
        logout();
      }
    } catch (e) {
      logout();
    }
    setLoading(false);
  };

  const login = async (email, password) => {
    let res;
    try {
      res = await fetch('/api/auth/login', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ email, password })
      });
    } catch {
      throw new Error('Cannot reach API. Is the backend running on port 8000?');
    }
    const data = await parseJsonResponse(res, 'Login failed');
    if (!res.ok) throw new Error(formatErrorDetail(data.detail, 'Login failed'));
    
    localStorage.setItem('token', data.access_token);
    setToken(data.access_token);
    setUser(data.user);
    setOrganization(data.organization);
    return data;
  };

  const signup = async (formData) => {
    let res;
    try {
      res = await fetch('/api/auth/signup', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(formData)
      });
    } catch {
      throw new Error('Cannot reach API. Is the backend running on port 8000?');
    }
    const data = await parseJsonResponse(res, 'Signup failed');
    if (!res.ok) throw new Error(formatErrorDetail(data.detail, 'Signup failed'));
    
    localStorage.setItem('token', data.access_token);
    setToken(data.access_token);
    setUser(data.user);
    setOrganization(data.organization);
    return data;
  };

  const logout = () => {
    localStorage.removeItem('token');
    setToken(null);
    setUser(null);
    setOrganization(null);
  };

  return (
    <AuthContext.Provider value={{ user, organization, token, loading, login, signup, logout }}>
      {children}
    </AuthContext.Provider>
  );
}

export const useAuth = () => useContext(AuthContext);
