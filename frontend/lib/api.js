// Thin client for the RepoSense AI backend.
//
// The backend has two different error shapes, and the UI needs a readable
// message from both:
//   * our own errors  -> { error, message, detail }
//   * FastAPI 422s    -> { detail: [ { loc, msg, type }, ... ] }

const API_URL = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

function readError(status, data) {
  if (data && typeof data.message === "string") {
    return data.detail ? `${data.message} (${data.detail})` : data.message;
  }
  if (Array.isArray(data?.detail)) {
    return data.detail.map((d) => d.msg).join("; ");
  }
  if (typeof data?.detail === "string") return data.detail;
  return `Request failed with status ${status}`;
}

async function request(path, { method = "GET", body } = {}) {
  let response;
  try {
    response = await fetch(`${API_URL}${path}`, {
      method,
      headers: body ? { "Content-Type": "application/json" } : undefined,
      body: body ? JSON.stringify(body) : undefined,
    });
  } catch {
    // fetch only rejects on network-level failures, which almost always means
    // the backend is not running - worth saying so explicitly.
    throw new Error(
      `Cannot reach the backend at ${API_URL}. Is the server running?`
    );
  }

  const data = await response.json().catch(() => null);
  if (!response.ok) throw new Error(readError(response.status, data));
  return data;
}

export function getHealth() {
  return request("/health");
}

export function analyzeRepository({ repositoryUrl, githubToken }) {
  return request("/api/repository/analyze", {
    method: "POST",
    body: {
      repository_url: repositoryUrl,
      // Omit rather than send an empty string, so the backend falls back to
      // GITHUB_TOKEN from .env.
      ...(githubToken ? { github_token: githubToken } : {}),
      auto_index: true,
    },
  });
}

export function askQuestion({ repositoryId, question }) {
  return request("/api/chat", {
    method: "POST",
    body: { repository_id: repositoryId, question },
  });
}

export function searchRepository({ repositoryId, query, topK = 5 }) {
  return request("/api/search", {
    method: "POST",
    body: { repository_id: repositoryId, query, top_k: topK },
  });
}

export { API_URL };
