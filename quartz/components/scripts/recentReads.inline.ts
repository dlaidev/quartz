const FEED_URL = "https://reading.mlai.blog/api/feed"

document.addEventListener("nav", async () => {
  const container = document.querySelector(".recent-reads-list")
  if (!container) return

  const limit = container.getAttribute("data-limit") || "5"

  container.innerHTML = `<p class="reads-loading">Loading recent reads...</p>`

  try {
    const res = await fetch(`${FEED_URL}?limit=${limit}`)
    if (!res.ok) throw new Error(`HTTP ${res.status}`)
    const items: {
      id: string
      title: string
      url: string
      source_type: string
      parent_category: string
      author?: string
      created_at: string
    }[] = await res.json()

    if (items.length === 0) {
      container.innerHTML = `<p class="reads-loading">No recent reads yet.</p>`
      return
    }

    const listHtml = items
      .map((item) => {
        const date = new Date(item.created_at).toLocaleDateString("en-US", {
          month: "short",
          day: "numeric",
        })
        const author = item.author ? ` · ${item.author}` : ""
        // Link to internal reading notes page, not the external source
        const noteUrl = `/reading/${item.id}`
        return `<li>
          <p class="read-title"><a href="${noteUrl}">${item.title}</a></p>
          <p class="read-meta">${date}${author}</p>
          <span class="read-category">${item.parent_category}</span>
        </li>`
      })
      .join("")

    container.innerHTML = `<ul class="reads-ul">${listHtml}</ul>`
  } catch (e) {
    container.innerHTML = `<p class="reads-error">Could not load recent reads.</p>`
  }
})
