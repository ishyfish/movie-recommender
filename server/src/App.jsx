import { useState, useEffect } from 'react'
import './App.css'
import { getMovies, submitRating, getRecommendations, getSimilar } from './api'

function App() {
  const [movies, setMovies] = useState([])
  const [query, setQuery] = useState("")
  const [skip, setSkip] = useState(0)
  const [recs, setRecs] = useState(null)
  const [similar, setSimilar] = useState(null)
  const [userId, setUserId] = useState(1)
  const [ratings, setRatings] = useState({})

  useEffect(() => {
    getMovies({ q: query, skip, limit: 20 }).then(setMovies)
  }, [query, skip])

  useEffect(() => {
    getRecommendations(userId).then(setRecs)
  }, [userId])

  function rate(movieId, score) {
    setRatings(prev => ({ ...prev, [movieId]: score }))
    submitRating({ user_id: userId, movie_id: movieId, rating: score })
      .then(() => {
        console.log(`rated ${movieId}: ${score}`)
        getRecommendations(userId).then(setRecs)
      })
  }

  function showSimilar(movieId) {
    getSimilar(movieId).then(setSimilar)
  }

  return (
    <div className="app">
      <header className="header">
        <div className="brand">
          <span className="logo">🎬</span>
          <div>
            <h1>Movie Recommender</h1>
            <p className="tagline">Matrix factorization, from scratch</p>
          </div>
        </div>
        <label className="user-select">
          <span>User</span>
          <input
            type="number"
            min="1"
            value={userId}
            onChange={e => setUserId(Number(e.target.value))}
          />
        </label>
      </header>

      {recs && (
        <section className="panel">
          <div className="panel-head">
            <h2>{recs.personalized ? "Recommended for you" : "Popular picks"}</h2>
            <span className={`badge ${recs.personalized ? "badge-personal" : "badge-popular"}`}>
              {recs.personalized ? "Personalized" : "Cold start"}
            </span>
          </div>
          <div className="card-grid">
            {recs.movies.map((m, i) => (
              <div key={m.movie_id} className="movie-card">
                <span className="rank">{i + 1}</span>
                <span className="card-title">{m.title}</span>
              </div>
            ))}
          </div>
        </section>
      )}

      <section className="panel">
        <div className="panel-head">
          <h2>Browse</h2>
        </div>
        <input
          className="search"
          value={query}
          onChange={e => { setQuery(e.target.value); setSkip(0); }}
          placeholder="Search movies…"
        />
        <ul className="movie-list">
          {movies.map(m => (
            <li key={m.movie_id} className="movie-row">
              <span className="title">{m.title}</span>
              <span className="actions">
                <span className="stars">
                  {[5, 4, 3, 2, 1].map(score => (
                    <button
                      key={score}
                      className={score <= (ratings[m.movie_id] || 0) ? "star filled" : "star"}
                      title={`Rate ${score}`}
                      onClick={() => rate(m.movie_id, score)}
                    >
                      ★
                    </button>
                  ))}
                </span>
                <button className="ghost-btn" onClick={() => showSimilar(m.movie_id)}>
                  Similar
                </button>
              </span>
            </li>
          ))}
        </ul>
        <div className="pager">
          <button className="ghost-btn" onClick={() => setSkip(Math.max(0, skip - 20))} disabled={skip === 0}>
            ← Prev
          </button>
          <span className="page-num">Page {skip / 20 + 1}</span>
          <button className="ghost-btn" onClick={() => setSkip(skip + 20)}>Next →</button>
        </div>
      </section>

      {similar && (
        <section className="panel">
          <div className="panel-head">
            <h2>Similar movies</h2>
          </div>
          <div className="card-grid">
            {similar.map(s => (
              <div key={s.movie_id} className="movie-card">
                <span className="card-title">{s.title}</span>
              </div>
            ))}
          </div>
        </section>
      )}
    </div>
  )
}

export default App
