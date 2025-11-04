import { useState, useEffect } from 'react'
import './App.css'

const API_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000'

function App() {
  const [systemName, setSystemName] = useState('Loading...')
  const [activeState, setActiveState] = useState(false)
  const [currentText, setCurrentText] = useState(null)
  const [activeEntryId, setActiveEntryId] = useState(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState(null)
  const [strokeData, setStrokeData] = useState(null)

  // Fetch system name and state on mount
  useEffect(() => {
    fetchSystemName()
    checkSystemState()
    
    // Poll system state every 2 seconds
    const interval = setInterval(() => {
      checkSystemState()
    }, 2000)
    
    return () => clearInterval(interval)
  }, [])
  
  // Fetch initial text when system becomes active (only once, no polling)
  useEffect(() => {
    if (activeState && !currentText && !loading) {
      fetchNextNeededData()
    }
  }, [activeState, currentText, loading])
  
  // Poll to check if text changed (when app submits, entry will change)
  useEffect(() => {
    if (!activeState || !currentText || loading) return
    
    // Poll every 2 seconds to check if entry changed (ID or text changed)
    const pollInterval = setInterval(async () => {
      if (activeState && !loading && currentText) {
        try {
          const response = await fetch(`${API_URL}/get_next_needed_data`)
          const data = await response.json()
          
          // Check if we got "no more data" message
          if (data.message && data.message === 'No more data to process') {
            setCurrentText(null)
            setActiveEntryId(null)
            setError('No more data to process')
            return
          }
          
          // If we got a different entry ID OR different text, update
          // (Text check handles case where ID might be same but entry was processed)
          if (data.id && data.entry_text && 
              (data.id !== activeEntryId || data.entry_text !== currentText)) {
            setCurrentText(data.entry_text)
            setActiveEntryId(data.id)
            setError(null) // Clear any previous errors
          }
        } catch (err) {
          // Silently fail - don't spam errors
          console.error('Error checking for text update:', err)
        }
      }
    }, 2000) // Check every 2 seconds
    
    return () => clearInterval(pollInterval)
  }, [activeState, currentText, loading, activeEntryId])

  // Fetch system name
  const fetchSystemName = async () => {
    try {
      const response = await fetch(`${API_URL}/get_system_name`)
      const data = await response.json()
      setSystemName(data.name || 'Handwriting AI')
      // Update document title
      document.title = data.name || 'Handwriting AI'
    } catch (err) {
      console.error('Error fetching system name:', err)
      setSystemName('Handwriting AI')
    }
  }

  // Check system state (only check, don't auto-fetch text)
  const checkSystemState = async () => {
    try {
      const response = await fetch(`${API_URL}/get_system_states`)
      const data = await response.json()
      setActiveState(data.active_state)
    } catch (err) {
      console.error('Error checking system state:', err)
      setError('Failed to connect to server')
    }
  }

  // Toggle system state
  const toggleSystemState = async (newState) => {
    setLoading(true)
    setError(null)
    try {
      const response = await fetch(`${API_URL}/change_system_states`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ active_state: newState }),
      })
      
      if (!response.ok) {
        throw new Error('Failed to change system state')
      }
      
      setActiveState(newState)
      
      // If turning on, fetch next needed data
      if (newState) {
        await fetchNextNeededData()
      } else {
        // If turning off, clear current data
        setCurrentText(null)
        setActiveEntryId(null)
        setStrokeData(null)
      }
    } catch (err) {
      console.error('Error toggling system state:', err)
      setError('Failed to change system state')
    } finally {
      setLoading(false)
    }
  }

  // Fetch next needed data
  const fetchNextNeededData = async () => {
    setLoading(true)
    setError(null)
    try {
      const response = await fetch(`${API_URL}/get_next_needed_data`)
      const data = await response.json()
      
      if (data.message && data.message === 'No more data to process') {
        setCurrentText(null)
        setActiveEntryId(null)
        setError('No more data to process')
      } else {
        setCurrentText(data.entry_text)
        setActiveEntryId(data.id)
        setStrokeData(null) // Reset stroke data when new entry is loaded
      }
    } catch (err) {
      console.error('Error fetching next needed data:', err)
      setError('Failed to fetch next data')
    } finally {
      setLoading(false)
    }
  }

  // Update InData with stroke data
  const updateInData = async (strokeDataToSend) => {
    if (!strokeDataToSend) {
      setError('No stroke data provided')
      return
    }

    setLoading(true)
    setError(null)
    try {
      const response = await fetch(`${API_URL}/update_InData`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ stroke_data: strokeDataToSend }),
      })
      
      if (!response.ok) {
        throw new Error('Failed to update InData')
      }
      
      // After successful update, fetch next needed data
      setCurrentText(null)
      setActiveEntryId(null)
      setStrokeData(null)
      // Wait a moment for backend to process, then fetch next text
      setTimeout(async () => {
        await fetchNextNeededData()
      }, 500)
    } catch (err) {
      console.error('Error updating InData:', err)
      setError('Failed to update InData')
    } finally {
      setLoading(false)
    }
  }

  // Handle stroke data input (placeholder - you'll need to integrate with your actual stroke capture)
  const handleStrokeDataInput = () => {
    // For now, using a placeholder. Replace this with actual stroke data capture
    const sampleStrokeData = {
      strokes: [],
      timestamp: Date.now()
    }
    setStrokeData(sampleStrokeData)
  }

  return (
    <div className="app-container">
      <header className="app-header">
        <h1>{systemName}</h1>
      </header>

      <main className="app-main">
        {error && (
          <div className="error-message">
            {error}
          </div>
        )}

        <div className="system-controls">
          <div className="toggle-section">
            <label className="toggle-label">
              System Active:
              <input
                type="checkbox"
                checked={activeState}
                onChange={(e) => toggleSystemState(e.target.checked)}
                disabled={loading}
                className="toggle-switch"
              />
            </label>
          </div>
        </div>

        {!activeState ? (
          <div className="inactive-message">
            <p>System is inactive. Turn on the system to start processing data.</p>
          </div>
        ) : (
          <div className="active-content">
            {loading && (
              <div className="loading-message">Loading...</div>
            )}

            {currentText && (
              <div className="text-display">
                <h2>Current Entry:</h2>
                <div className="text-content">{currentText}</div>
                <div className="entry-id">Entry ID: {activeEntryId}</div>
              </div>
            )}

            {currentText && (
              <div className="stroke-data-section">
                <h3>Stroke Data</h3>
                <button 
                  onClick={handleStrokeDataInput}
                  disabled={loading}
                  className="capture-button"
                >
                  Capture Stroke Data
                </button>
                
                {strokeData && (
                  <div className="stroke-data-display">
                    <pre>{JSON.stringify(strokeData, null, 2)}</pre>
                    <button
                      onClick={() => updateInData(strokeData)}
                      disabled={loading}
                      className="submit-button"
                    >
                      Submit Stroke Data
                    </button>
                  </div>
                )}
              </div>
            )}

            {!currentText && !loading && (
              <div className="no-data-message">
                <p>No data available. Waiting for next entry...</p>
              </div>
            )}
          </div>
        )}
      </main>
    </div>
  )
}

export default App
