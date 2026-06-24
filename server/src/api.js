const BASE = import.meta.env.VITE_API_URL

async function request(path, options) {
    const response = await fetch(`${BASE}${path}`, options)
    if (!response.ok){
        throw new Error(`API${response.status}`)
    }
    return response.json()
}

export async function getMovies({skip = 0, limit = 50, q } = {}) {
    const params = new URLSearchParams({skip, limit})
    if (q){
        params.set("q", q)
    }
    return request(`/movies?${params.toString()}`)
}

export async function getRecommendations(userId, n = 10){
    return request(`/recommendations/${userId}?n=${n}`)
}

export async function getSimilar(movieId, n = 10){
    return request(`/movies/similar/${movieId}?n=${n}`)
}

export async function submitRating(rating){
    return request(
        '/ratings', 
        {
            method: "POST",
            headers: { "Content-Type": "application/json" }, 
            body: JSON.stringify(rating)
        }
    )
}

