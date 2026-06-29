import { useState, useEffect, useEffectEvent } from 'react'
import './App.css'
import { getMovies } from './api'

function App() {
  const [movies, setMovies] = useState([])
  useEffect(() => {
    getMovies({ limit:20}).then(setMovies)
  }, [])

  return (
    <div>
      <h1>Movies</h1>
      <ul>
        {movies.map(m => (
          <li key={m.movie_id}>{m.title}</li>
        ))}
      </ul>
    </div>
  )
  
  
}



export default App
