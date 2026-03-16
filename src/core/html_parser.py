import re
import requests
from utils.deps import require
bs_available, bs_message = require("bs4", "BeautifulSoup (bs4)", optional=True)
if bs_available:
    from bs4 import BeautifulSoup  # type: ignore
from urllib.parse import urljoin, urlparse


class HTMLParser:
    def __init__(self, max_depth=1):
        self.max_depth = max_depth
        self.visited = set()

    def extract_text(self, html_content):
        if not bs_available:
            raise ImportError(bs_message)
        soup = BeautifulSoup(html_content, "html.parser")
        for script in soup(["script", "style"]):
            script.decompose()
        text = soup.get_text(separator="\n")
        text = re.sub(r"\n+", "\n", text)
        return text.strip()

    def crawl(self, url, depth=0):
        if not bs_available:
            raise ImportError(bs_message)
        if depth > self.max_depth or url in self.visited:
            return []
        self.visited.add(url)
        try:
            resp = requests.get(url, timeout=10)
            resp.raise_for_status()
        except Exception:
            return []
        text = self.extract_text(resp.text)
        results = [text] if text else []
        if depth == self.max_depth:
            return results
        soup = BeautifulSoup(resp.text, "html.parser")
        for link in soup.find_all("a", href=True):
            href = link["href"]
            absolute = urljoin(url, href)
            if urlparse(absolute).scheme in ("http", "https"):
                results.extend(self.crawl(absolute, depth + 1))
        return results
