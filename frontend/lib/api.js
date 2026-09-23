// Client for the RepoSense AI backend.
//
// The backend has two error shapes and the UI needs a readable message from
// both:
//   * our own errors -> { error, message, detail }
//   * FastAPI 422s   -> { detail: [ { loc, msg, type }, ... ] }

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
    // fetch only rejects on network-level failures, which in practice means
    // the backend is not running.
    throw new Error(
      `Cannot reach the backend at ${API_URL}. Is the server running?`
    );
  }

  if (response.status === 204) return null;

  const data = await response.json().catch(() => null);
  if (!response.ok) throw new Error(readError(response.status, data));
  return data;
}

// ---------------------------------------------------------------- system
export const getHealth = () => request("/health");

// ---------------------------------------------------------------- repos
export const analyzeRepository = ({ repositoryUrl, githubToken }) =>
  request("/api/repository/analyze", {
    method: "POST",
    body: {
      repository_url: repositoryUrl,
      // Omitted rather than sent empty, so the backend falls back to the
      // server-side GITHUB_TOKEN.
      ...(githubToken ? { github_token: githubToken } : {}),
      auto_index: true,
    },
  });

export const getRepositoryStatus = (repositoryId) =>
  request(
    `/api/repository/status${
      repositoryId ? `?repository_id=${encodeURIComponent(repositoryId)}` : ""
    }`
  );

// ---------------------------------------------------------------- search
export const searchRepository = ({ repositoryId, query, topK = 5 }) =>
  request("/api/search", {
    method: "POST",
    body: { repository_id: repositoryId, query, top_k: topK },
  });

// ---------------------------------------------------------- conversations
export const listConversations = (repositoryId) =>
  request(
    `/api/conversations${
      repositoryId ? `?repository_id=${encodeURIComponent(repositoryId)}` : ""
    }`
  );

export const getConversation = (id) => request(`/api/conversations/${id}`);

export const startConversation = ({ repositoryId, question }) =>
  request("/api/conversations", {
    method: "POST",
    body: { repository_id: repositoryId, question },
  });

export const sendMessage = ({ conversationId, question }) =>
  request(`/api/conversations/${conversationId}/messages`, {
    method: "POST",
    body: { question },
  });

export const renameConversation = ({ id, title }) =>
  request(`/api/conversations/${id}`, { method: "PATCH", body: { title } });

export const deleteConversation = (id) =>
  request(`/api/conversations/${id}`, { method: "DELETE" });

// ------------------------------------------------------------- analytics
export const getAnalytics = () => request("/api/analytics/overview");

export { API_URL };
