from __future__ import annotations

import json
import re
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import requests


@dataclass
class WikiPage:
    title: str
    summary: str
    content: str
    url: str = ""


class WikipediaClient:
    """Online Wikipedia client with an optional user-owned SQLite cache."""

    def __init__(self, language: str = "en", cache_path: str | None = None, timeout: float = 30.0):
        self.language = language
        self.endpoint = f"https://{language}.wikipedia.org/w/api.php"
        self.timeout = timeout
        self.cache_path = Path(cache_path).expanduser() if cache_path else None
        if self.cache_path:
            self.cache_path.parent.mkdir(parents=True, exist_ok=True)
            with sqlite3.connect(self.cache_path) as db:
                db.execute("CREATE TABLE IF NOT EXISTS pages (title TEXT PRIMARY KEY, payload TEXT)")

    def _cached(self, title: str) -> WikiPage | None:
        if not self.cache_path:
            return None
        with sqlite3.connect(self.cache_path) as db:
            row = db.execute("SELECT payload FROM pages WHERE title = ?", (title,)).fetchone()
        if not row:
            return None
        return WikiPage(**json.loads(row[0]))

    def _store(self, page: WikiPage) -> None:
        if not self.cache_path:
            return
        with sqlite3.connect(self.cache_path) as db:
            db.execute("INSERT OR REPLACE INTO pages(title, payload) VALUES (?, ?)",
                       (page.title, json.dumps(page.__dict__, ensure_ascii=False)))

    def page(self, title: str) -> WikiPage | None:
        title = title.strip()
        if not title:
            return None
        cached = self._cached(title)
        if cached:
            return cached
        params = {"action": "query", "prop": "extracts|info", "explaintext": 1,
                  "inprop": "url", "titles": title, "format": "json", "redirects": 1}
        response = requests.get(self.endpoint, params=params, timeout=self.timeout)
        response.raise_for_status()
        pages = response.json().get("query", {}).get("pages", {})
        page_data = next(iter(pages.values()), {})
        if "missing" in page_data:
            return None
        content = str(page_data.get("extract", ""))
        page = WikiPage(title=str(page_data.get("title", title)), summary=content[:1200],
                        content=content, url=str(page_data.get("fullurl", "")))
        self._store(page)
        return page

    def search(self, query: str, limit: int = 5) -> list[str]:
        params = {"action": "query", "list": "search", "srsearch": query,
                  "srlimit": limit, "format": "json"}
        response = requests.get(self.endpoint, params=params, timeout=self.timeout)
        response.raise_for_status()
        return [str(item.get("title", "")) for item in response.json().get("query", {}).get("search", [])]


class WikiReactEnvironment:
    def __init__(self, client: WikipediaClient, max_steps: int = 10, lookup_window: int = 2000):
        self.client = client
        self.max_steps = max_steps
        self.lookup_window = lookup_window
        self.steps = 0
        self.current_page: WikiPage | None = None
        self.answer = ""
        self.terminated = False

    def reset(self) -> None:
        self.steps = 0
        self.current_page = None
        self.answer = ""
        self.terminated = False

    def search(self, query: str) -> str:
        page = self.client.page(query)
        if page:
            self.current_page = page
            content = page.content or page.summary
            return (
                "Search hit:\n"
                f"- title: {page.title}\n"
                f"- content: {self._compact_text(content, max_chars=1200)}"
            )
        titles = self.client.search(query, limit=5)
        if not titles:
            return f"Search failed: no page found for query {query!r}."
        return (
            f"No exact page found for query {query!r}.\n"
            "You may refer to these candidate pages:\n"
            "- " + "\n- ".join(titles[:5])
        )

    def lookup(self, keyword: str) -> str:
        if not self.current_page:
            return "Lookup failed: use Search first."
        keyword = keyword.strip()
        if not keyword:
            return "Lookup failed: keyword is empty."
        for source_name, text in (("summary", self.current_page.summary),
                                  ("content", self.current_page.content)):
            snippet = self._extract_lookup_snippet(text, keyword)
            if snippet:
                return (
                    f"Lookup in page {self.current_page.title!r} for keyword {keyword!r} "
                    f"({source_name}): {snippet}"
                )
        return f"Lookup found no match for keyword {keyword!r} in current page {self.current_page.title!r}."

    def _extract_lookup_snippet(self, text: str, keyword: str) -> str:
        normalized_text = str(text or "").strip()
        normalized_keyword = str(keyword or "").strip()
        if not normalized_text or not normalized_keyword:
            return ""
        index = normalized_text.casefold().find(normalized_keyword.casefold())
        if index < 0:
            return ""
        start = max(0, index - self.lookup_window // 2)
        end = min(len(normalized_text), index + len(normalized_keyword) + self.lookup_window // 2)
        snippet = normalized_text[start:end].strip()
        if start > 0:
            snippet = f"... {snippet}"
        if end < len(normalized_text):
            snippet = f"{snippet} ..."
        return self._compact_text(snippet, max_chars=self.lookup_window + 40)

    @staticmethod
    def _compact_text(text: str, *, max_chars: int) -> str:
        normalized = re.sub(r"\s+", " ", str(text or "")).strip()
        if len(normalized) <= max_chars:
            return normalized
        keep = max(16, max_chars - len(" ... [truncated]"))
        return f"{normalized[:keep].rstrip()} ... [truncated]"

    def step(self, action: str) -> str:
        self.steps += 1
        text = action.strip()
        if text.lower().startswith("search[") and text.endswith("]"):
            return self.search(text[7:-1])
        if text.lower().startswith("lookup[") and text.endswith("]"):
            return self.lookup(text[7:-1])
        if text.lower().startswith("finish[") and text.endswith("]"):
            self.answer = text[7:-1].strip()
            self.terminated = True
            return "Finished."
        if self.steps >= self.max_steps:
            self.terminated = True
        return "Invalid action. Use Search[query], Lookup[keyword], or Finish[answer]."
