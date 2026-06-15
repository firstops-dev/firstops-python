"""FirstOps management client for programmatic agent and connection CRUD."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import httpx


@dataclass
class Agent:
    """An agent principal."""

    id: str
    tenant_id: str
    name: str
    reference_id: str
    metadata: dict[str, str]
    created_at: int
    token: str  # fo_agent_<id> — used in Authorization header
    private_key: str | None = None  # PEM, only set on creation


@dataclass
class Connection:
    """A registered MCP connection."""

    id: str
    tenant_id: str
    principal_id: str
    name: str
    upstream_url: str
    status: str
    created_at: int


@dataclass
class ParamDefinition:
    """A single parameter required to configure a template connection."""

    key: str           # placeholder key, e.g. "api_token"
    display_name: str  # human-readable label
    description: str   # help text
    required: bool
    secret: bool       # if true, treat the value as sensitive
    maps_to: str       # "HEADER", "QUERY_PARAM", or "URL_PLACEHOLDER"


@dataclass
class ServerTemplate:
    """A FirstOps catalog entry describing an MCP server integration."""

    id: str
    name: str
    description: str
    category: str
    upstream_url_template: str
    required_params: list[ParamDefinition]
    tenant_configured: bool
    user_connected: bool
    existing_connection_id: str
    auth_type: str
    transport_type: str
    oauth_setup_guide: str
    requires_org_config: bool
    raw: dict[str, Any]  # full proto-JSON for fields the SDK doesn't model


def _parse_template(entry: dict[str, Any]) -> ServerTemplate:
    """Parse a ServerTemplateWithStatus proto-JSON entry.

    The catalog RPC returns ``ServerTemplateWithStatus``, a wrapper with
    the actual ``ServerTemplate`` nested under ``template`` plus three
    overlay flags (``tenant_configured``, ``user_connected``,
    ``existing_connection_id``). We flatten that into a single dataclass.

    When the backend returns a bare ``ServerTemplate`` (no wrapper), we
    fall back to reading fields off the entry itself.
    """
    inner = (
        entry.get("template")
        if isinstance(entry.get("template"), dict)
        else entry
    )

    params_raw = inner.get("required_params") or []
    params: list[ParamDefinition] = []
    for p in params_raw:
        params.append(
            ParamDefinition(
                key=p.get("key", ""),
                display_name=p.get("display_name", ""),
                description=p.get("description", ""),
                required=p.get("required", False),
                secret=p.get("secret", False),
                maps_to=p.get("maps_to", ""),
            )
        )

    # oauth_setup_guide is a nested message — stringify to something usable.
    oauth_guide = inner.get("oauth_setup_guide")
    if isinstance(oauth_guide, dict):
        oauth_guide_str = oauth_guide.get("instructions", "") or ""
    elif isinstance(oauth_guide, str):
        oauth_guide_str = oauth_guide
    else:
        oauth_guide_str = ""

    return ServerTemplate(
        id=inner.get("id", ""),
        name=inner.get("name", ""),
        description=inner.get("description", ""),
        category=inner.get("category", ""),
        upstream_url_template=inner.get("upstream_url", ""),
        required_params=params,
        tenant_configured=entry.get("tenant_configured", False),
        user_connected=entry.get("user_connected", False),
        existing_connection_id=entry.get("existing_connection_id", ""),
        auth_type=inner.get("auth_method", "") or inner.get("auth_type", ""),
        transport_type=inner.get("transport_type", ""),
        oauth_setup_guide=oauth_guide_str,
        requires_org_config=inner.get("requires_org_config", False),
        raw=entry,
    )


class FirstOpsError(Exception):
    """Raised when the FirstOps API returns an error."""

    def __init__(self, status_code: int, message: str):
        self.status_code = status_code
        self.message = message
        super().__init__(f"{message} (HTTP {status_code})")


class _AgentsResource:
    def __init__(self, client: FirstOps):
        self._client = client

    def create(self, name: str) -> Agent:
        """Create a new agent principal with a DPoP keypair."""
        resp = self._client._request("POST", "/api/v1/sdk/agents", json={"name": name})
        agent_data = resp["agent"]
        return Agent(
            id=agent_data["id"],
            tenant_id=agent_data.get("tenant_id", ""),
            name=agent_data.get("name", "") or agent_data.get("metadata", {}).get("name", ""),
            reference_id=agent_data.get("reference_id", ""),
            metadata=agent_data.get("metadata", {}),
            created_at=agent_data.get("created_at", 0),
            token=f"fo_agent_{agent_data['id']}",
            private_key=resp.get("private_key"),
        )

    def list(self) -> list[Agent]:
        """List all agent principals in the tenant."""
        resp = self._client._request("GET", "/api/v1/sdk/agents")
        agents = []
        for a in resp.get("agents", []):
            agents.append(
                Agent(
                    id=a["id"],
                    tenant_id=a.get("tenant_id", ""),
                    name=a.get("name", "") or a.get("metadata", {}).get("name", ""),
                    reference_id=a.get("reference_id", ""),
                    metadata=a.get("metadata", {}),
                    created_at=a.get("created_at", 0),
                    token=f"fo_agent_{a['id']}",
                )
            )
        return agents

    def get(self, agent_id: str) -> Agent:
        """Get a single agent by ID."""
        resp = self._client._request("GET", f"/api/v1/sdk/agents/{agent_id}")
        a = resp["agent"]
        return Agent(
            id=a["id"],
            tenant_id=a.get("tenant_id", ""),
            name=a.get("name", "") or a.get("metadata", {}).get("name", ""),
            reference_id=a.get("reference_id", ""),
            metadata=a.get("metadata", {}),
            created_at=a.get("created_at", 0),
            token=f"fo_agent_{a['id']}",
        )

    def delete(self, agent_id: str) -> None:
        """Delete an agent principal."""
        self._client._request("DELETE", f"/api/v1/sdk/agents/{agent_id}")


class _ConnectionsResource:
    def __init__(self, client: FirstOps):
        self._client = client

    def register(
        self,
        *,
        principal_id: str,
        # Template-based registration (preferred for catalog entries)
        template_id: str = "",
        user_params: dict[str, str] | None = None,
        # Raw registration (when not using a template)
        name: str = "",
        upstream_url: str = "",
        auth_type: str = "",
        transport_type: str = "",
        upstream_headers: dict[str, str] | None = None,
        upstream_query_params: dict[str, str] | None = None,
        source: str = "sdk",
    ) -> Connection:
        """Register an MCP connection for an agent.

        Two modes:

        1. **Template-based** (preferred): pass `template_id` from the
           FirstOps catalog plus `user_params` for the required placeholders.
           The backend resolves the upstream URL, pre-fills auth_type and
           transport_type from the template, and maps params to headers or
           query params per the template's `ParamDefinition.maps_to` rules.

        2. **Raw**: pass `name` and `upstream_url` directly. Use this only
           when registering a custom MCP server that isn't in the catalog.

        Example (template):
            conn = client.connections.register(
                principal_id=agent.id,
                template_id="github-pat",
                user_params={"api_token": "ghp_..."},
            )

        Example (raw):
            conn = client.connections.register(
                principal_id=agent.id,
                name="my-custom-mcp",
                upstream_url="https://mcp.internal.corp/sse",
            )
        """
        body: dict[str, Any] = {
            "principal_id": principal_id,
            "source": source,
        }

        if template_id:
            body["template_id"] = template_id
            if user_params:
                body["user_params"] = user_params
            # Template flow still requires a name for the connection record.
            # If the caller doesn't supply one, derive from the template ID.
            body["name"] = name or template_id
            # upstream_url is resolved server-side from the template.
            body["upstream_url"] = upstream_url  # empty is fine here
        else:
            if not name or not upstream_url:
                raise ValueError(
                    "register() requires either template_id or "
                    "both name and upstream_url"
                )
            body["name"] = name
            body["upstream_url"] = upstream_url

        if auth_type:
            body["auth_type"] = auth_type
        if transport_type:
            body["transport_type"] = transport_type
        if upstream_headers:
            body["upstream_headers"] = upstream_headers
        if upstream_query_params:
            body["upstream_query_params"] = upstream_query_params

        resp = self._client._request(
            "POST", "/api/v1/sdk/connections/register", json=body
        )
        c = resp["connection"]
        return Connection(
            id=c["id"],
            tenant_id=c.get("tenant_id", ""),
            principal_id=c.get("principal_id", ""),
            name=c.get("mcp_server_name", "") or c.get("name", ""),
            upstream_url=c.get("upstream_url", ""),
            status=c.get("status", ""),
            created_at=c.get("created_at", 0),
        )

    def list(self, *, principal_id: str | None = None) -> list[Connection]:
        """List connections, optionally filtered by principal."""
        params: dict[str, str] = {}
        if principal_id:
            params["principal_id"] = principal_id

        resp = self._client._request("GET", "/api/v1/sdk/connections", params=params)
        connections = []
        for c in resp.get("connections", []):
            connections.append(
                Connection(
                    id=c["id"],
                    tenant_id=c.get("tenant_id", ""),
                    principal_id=c.get("principal_id", ""),
                    name=c.get("mcp_server_name", "") or c.get("name", ""),
                    upstream_url=c.get("upstream_url", ""),
                    status=c.get("status", ""),
                    created_at=c.get("created_at", 0),
                )
            )
        return connections

    def delete(self, connection_id: str) -> None:
        """Delete a connection."""
        self._client._request("DELETE", f"/api/v1/sdk/connections/{connection_id}")


class _CatalogResource:
    def __init__(self, client: FirstOps):
        self._client = client

    def list(
        self,
        *,
        principal_id: str | None = None,
        search: str | None = None,
        category: str | None = None,
    ) -> list[ServerTemplate]:
        """List MCP server templates in the FirstOps catalog.

        When `principal_id` is supplied, the response includes
        `user_connected` and `existing_connection_id` overlays scoped to
        that agent — useful when rendering "Connect" vs "Already connected"
        states in a customer's UI.
        """
        params: dict[str, str] = {}
        if principal_id:
            params["principal_id"] = principal_id
        if search:
            params["search"] = search
        if category:
            params["category"] = category

        resp = self._client._request(
            "GET", "/api/v1/catalog/templates", params=params
        )
        return [_parse_template(t) for t in resp.get("templates", [])]

    def get(
        self, template_id: str, *, principal_id: str | None = None
    ) -> ServerTemplate:
        """Get a single catalog template by ID."""
        params: dict[str, str] = {}
        if principal_id:
            params["principal_id"] = principal_id

        resp = self._client._request(
            "GET", f"/api/v1/catalog/templates/{template_id}", params=params
        )
        return _parse_template(resp["template"])


class FirstOps:
    """Management client for the FirstOps API.

    Usage:
        client = FirstOps(api_key="fo_key_...")
        agent = client.agents.create(name="my-bot")
        client.connections.register(
            principal_id=agent.id,
            name="slack",
            upstream_url="https://mcp.slack.com/sse",
        )
    """

    def __init__(
        self,
        api_key: str,
        base_url: str = "https://api.firstops.dev",
        timeout: float = 30.0,
    ):
        if not api_key or not api_key.startswith("fo_key_"):
            raise ValueError("api_key must start with 'fo_key_'")

        self._base_url = base_url.rstrip("/")
        self._api_key = api_key
        self._http = httpx.Client(
            timeout=timeout,
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
        )
        self.agents = _AgentsResource(self)
        self.connections = _ConnectionsResource(self)
        self.catalog = _CatalogResource(self)

    def _request(
        self,
        method: str,
        path: str,
        *,
        json: dict[str, Any] | None = None,
        params: dict[str, str] | None = None,
    ) -> dict[str, Any]:
        resp = self._http.request(
            method,
            f"{self._base_url}{path}",
            json=json,
            params=params,
        )
        if resp.status_code >= 400:
            try:
                body = resp.json()
                msg = body.get("error", f"HTTP {resp.status_code}")
            except Exception:
                msg = f"HTTP {resp.status_code}"
            raise FirstOpsError(resp.status_code, msg)

        if resp.status_code == 204 or not resp.content:
            return {}
        return resp.json()

    def close(self) -> None:
        """Close the underlying HTTP client."""
        self._http.close()

    def __enter__(self) -> FirstOps:
        return self

    def __exit__(self, *args: Any) -> None:
        self.close()
