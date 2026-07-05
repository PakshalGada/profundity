"""
Codeforces Scraper – parallel scraping of ALL problems with statements,
editorials, and solutions.

Usage:
    # Scrape everything with 8 parallel workers:
    python codeforces.py --all --workers 8

    # Scrape with filters:
    python codeforces.py --all --min-rating 800 --max-rating 1600

    # Scrape specific problems:
    python codeforces.py CF-1560A CF-4B

    # Light mode (no editorial/solutions, much faster):
    python codeforces.py --all --light --workers 12

    # Test with a small batch:
    python codeforces.py --all --max 10

Output is written to data/codeforces_problems.jsonl (one JSON object per line).
"""

import json
import os
import random
import requests
import cloudscraper
from bs4 import BeautifulSoup
import time
import re
import logging
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Optional
from pathlib import Path

logger = logging.getLogger(__name__)

BASE_URL = "https://codeforces.com"
API_URL = f"{BASE_URL}/api"
MAX_RETRIES = 3
RETRY_BACKOFF = 2.0

# Defaults
DEFAULT_WORKERS = 4
DEFAULT_RPS = 4.0            # requests per second (global across all workers)

DATA_DIR = Path(__file__).resolve().parent / "data"
DEFAULT_OUTPUT = DATA_DIR / "codeforces_problems.jsonl"
DEFAULT_CHECKPOINT = DATA_DIR / "codeforces_checkpoint.txt"


# ======================================================================
# Token-bucket rate limiter (thread-safe)
# ======================================================================
class RateLimiter:
    """
    Token-bucket rate limiter shared across all threads.
    Ensures we never exceed `rate` requests per second globally.
    """

    def __init__(self, rate: float):
        self.rate = rate                    # tokens per second
        self.capacity = max(rate, 1.0)      # burst capacity
        self.tokens = self.capacity
        self.last = time.monotonic()
        self._lock = threading.Lock()

    def acquire(self):
        """Block until a token is available."""
        while True:
            with self._lock:
                now = time.monotonic()
                elapsed = now - self.last
                self.tokens = min(self.capacity,
                                  self.tokens + elapsed * self.rate)
                self.last = now
                if self.tokens >= 1.0:
                    self.tokens -= 1.0
                    return
            # Not enough tokens – sleep a short interval and retry
            time.sleep(1.0 / self.rate)


# ======================================================================
# Scraper
# ======================================================================
class CodeforcesScraper:
    def __init__(self, rps: float = DEFAULT_RPS,
                 cookies: Optional[dict] = None):
        # Default browser for cloudscraper
        browser_kwargs = {
            'browser': 'chrome',
            'platform': 'windows',
            'desktop': True
        }
        
        # If cookies.json contains a User-Agent, use it explicitly instead of randomizing
        custom_ua = None
        if cookies and "User-Agent" in cookies:
            custom_ua = cookies.pop("User-Agent")
            
        self.session = cloudscraper.create_scraper(browser=browser_kwargs)
        if custom_ua:
            self.session.headers.update({"User-Agent": custom_ua})

        if cookies:
            self.session.cookies.update(cookies)

        # Shared rate limiter
        self.limiter = RateLimiter(rps)

        # Thread-safe caches
        self._editorial_url_cache: dict[str, Optional[str]] = {}
        self._editorial_cache_lock = threading.Lock()
        self._editorial_fetch_locks: dict[str, threading.Lock] = {}
        self._editorial_fetch_locks_lock = threading.Lock()

        self._api_ratings: dict[str, Optional[int]] = {}

        # File I/O lock
        self._file_lock = threading.Lock()

        # Progress counter
        self._progress_lock = threading.Lock()
        self._done_count = 0
        self._error_count = 0

    def set_cookies(self, cookies: dict):
        self.session.cookies.update(cookies)

    # ------------------------------------------------------------------
    # HTTP helpers with retry + rate limiting
    # ------------------------------------------------------------------
    def _get(self, url: str, retries: int = MAX_RETRIES) -> requests.Response:
        self.limiter.acquire()
        for attempt in range(retries):
            try:
                logger.debug("GET %s (attempt %d)", url, attempt + 1)
                resp = self.session.get(url, timeout=30)
                if resp.status_code == 403:
                    logger.warning("HTTP 403 on %s – skipping", url)
                    resp.raise_for_status()
                if resp.status_code == 429:
                    wait = RETRY_BACKOFF ** (attempt + 1)
                    logger.warning("Rate limited on %s, waiting %.1fs ...",
                                   url, wait)
                    time.sleep(wait)
                    self.limiter.acquire()
                    continue
                resp.raise_for_status()
                return resp
            except requests.RequestException as exc:
                if attempt == retries - 1:
                    raise
                wait = RETRY_BACKOFF ** (attempt + 1)
                logger.warning("Request failed (%s), retry in %.1fs ...",
                               exc, wait)
                time.sleep(wait)
                self.limiter.acquire()
        raise RuntimeError(f"Exhausted retries for {url}")

    def _api_get(self, endpoint: str, params: dict | None = None) -> dict:
        """Call the Codeforces API (with retry + rate limit)."""
        url = f"{API_URL}/{endpoint}"
        self.limiter.acquire()
        for attempt in range(MAX_RETRIES):
            try:
                resp = self.session.get(url, params=params, timeout=60)
                if resp.status_code == 403:
                    logger.warning("HTTP 403 on %s – skipping", url)
                    resp.raise_for_status()
                if resp.status_code == 429:
                    wait = RETRY_BACKOFF ** (attempt + 1)
                    logger.warning("API rate limited, waiting %.1fs ...", wait)
                    time.sleep(wait)
                    self.limiter.acquire()
                    continue
                resp.raise_for_status()
                data = resp.json()
                if data.get("status") != "OK":
                    raise RuntimeError(f"API error: {data}")
                return data["result"]
            except requests.RequestException as exc:
                if attempt == MAX_RETRIES - 1:
                    raise
                wait = RETRY_BACKOFF ** (attempt + 1)
                logger.warning("API request failed (%s), retry in %.1fs ...",
                               exc, wait)
                time.sleep(wait)
                self.limiter.acquire()
        raise RuntimeError(f"Exhausted retries for {url}")

    # ------------------------------------------------------------------
    # ID helpers
    # ------------------------------------------------------------------
    @staticmethod
    def parse_problem_id(problem_id: str) -> tuple[str, str]:
        """Parse 'CF-1560A' → ('1560', 'A')."""
        match = re.fullmatch(r'CF-(\d+)([A-Za-z]\d*)', problem_id)
        if not match:
            raise ValueError(
                f"Invalid problem ID '{problem_id}'. Expected CF-<contestId><index> "
                f"(e.g. CF-1560A, CF-4B)."
            )
        return match.group(1), match.group(2).upper()

    @staticmethod
    def make_problem_id(contest_id, index) -> str:
        return f"CF-{contest_id}{index}"

    @staticmethod
    def problem_url(contest_id: str, index: str) -> str:
        return f"{BASE_URL}/problemset/problem/{contest_id}/{index}"

    # ------------------------------------------------------------------
    # API: fetch full problem list + ratings
    # ------------------------------------------------------------------
    def fetch_problem_list(self) -> list[dict]:
        """Return all problems from the Codeforces API."""
        logger.info("Fetching problem list from Codeforces API ...")
        result = self._api_get("problemset.problems")
        problems = result["problems"]

        for p in problems:
            cid = p.get("contestId")
            idx = p.get("index")
            if cid and idx:
                pid = self.make_problem_id(cid, idx)
                self._api_ratings[pid] = p.get("rating")

        logger.info("API returned %d problems", len(problems))
        return problems

    def _get_api_rating(self, problem_id: str) -> Optional[int]:
        return self._api_ratings.get(problem_id)

    # ------------------------------------------------------------------
    # Problem statement extraction
    # ------------------------------------------------------------------
    def _extract_problem_statement(self, soup: BeautifulSoup) -> str:
        ps = soup.find("div", class_="problem-statement")
        if ps is None:
            return ""
        return ps.get_text("\n", strip=True)

    def _extract_time_limit(self, soup: BeautifulSoup) -> str:
        div = soup.find("div", class_="time-limit")
        if div:
            text = div.get_text(" ", strip=True)
            text = re.sub(r'^time limit per test\s*', '', text, flags=re.I)
            return text.strip() or div.get_text(" ", strip=True)
        return ""

    def _extract_difficulty(self, soup: BeautifulSoup) -> Optional[int]:
        for tag in soup.find_all("span", class_="tag-box"):
            title = tag.get("title", "")
            if "difficulty" in title.lower():
                text = tag.get_text(strip=True)
                nums = re.findall(r'\d+', text)
                if nums:
                    return int(nums[0])
        return None

    # ------------------------------------------------------------------
    # Editorial: discover URL from the contest page (thread-safe)
    # ------------------------------------------------------------------
    def _get_editorial_fetch_lock(self, contest_id: str) -> threading.Lock:
        """Get a per-contest lock so only one thread fetches the editorial URL."""
        with self._editorial_fetch_locks_lock:
            if contest_id not in self._editorial_fetch_locks:
                self._editorial_fetch_locks[contest_id] = threading.Lock()
            return self._editorial_fetch_locks[contest_id]

    def _find_editorial_url(self, contest_id: str) -> Optional[str]:
        """Find the editorial blog entry URL for a contest (thread-safe)."""
        # Fast path: check cache without lock
        with self._editorial_cache_lock:
            if contest_id in self._editorial_url_cache:
                return self._editorial_url_cache[contest_id]

        # Slow path: only one thread per contest does the actual fetch
        fetch_lock = self._get_editorial_fetch_lock(contest_id)
        with fetch_lock:
            # Double-check after acquiring lock
            with self._editorial_cache_lock:
                if contest_id in self._editorial_url_cache:
                    return self._editorial_url_cache[contest_id]

            # Actually fetch
            editorial_url = self._fetch_editorial_url(contest_id)
            with self._editorial_cache_lock:
                self._editorial_url_cache[contest_id] = editorial_url
            return editorial_url

    def _fetch_editorial_url(self, contest_id: str) -> Optional[str]:
        """Fetch the editorial URL from the contest page."""
        contest_url = f"{BASE_URL}/contest/{contest_id}"
        try:
            resp = self._get(contest_url)
            soup = BeautifulSoup(resp.text, "html.parser")

            for a in soup.find_all("a", href=True):
                txt = a.get_text(strip=True).lower()
                href = a["href"]
                if re.search(r'\btutorial\b|\beditorial\b', txt):
                    if href.startswith("/"):
                        href = BASE_URL + href
                    logger.debug("Editorial for contest %s: %s",
                                 contest_id, href)
                    return href

            for a in soup.find_all("a", href=True):
                href = a["href"]
                if "/blog/entry/" in href:
                    txt = a.get_text(strip=True).lower()
                    if re.search(r'tutorial|editorial|разбор', txt):
                        full = href if href.startswith("http") else BASE_URL + href
                        return full
        except requests.RequestException:
            logger.warning("Failed to fetch contest page %s", contest_url)

        return None

    # ------------------------------------------------------------------
    # Editorial text extraction from the blog page
    # ------------------------------------------------------------------
    def _scrape_editorial_text(self, editorial_url: str,
                               contest_id: str, index: str) -> str:
        try:
            resp = self._get(editorial_url)
        except requests.RequestException:
            logger.warning("Failed to fetch editorial at %s", editorial_url)
            return ""

        soup = BeautifulSoup(resp.text, "html.parser")

        # Check for dynamic tutorial loading first
        target_code = f"{contest_id}{index}"
        tutorial_div = soup.find("div", class_="problemTutorial", problemcode=target_code)
        if tutorial_div:
            csrf_meta = soup.find("meta", {"name": "X-Csrf-Token"})
            if csrf_meta:
                csrf_token = csrf_meta.get("content")
                try:
                    self.limiter.acquire()
                    tut_resp = self.session.post(
                        f"{BASE_URL}/data/problemTutorial",
                        data={"problemCode": target_code, "csrf_token": csrf_token},
                        timeout=30
                    )
                    tut_data = tut_resp.json()
                    if tut_data.get("success") == "true":
                        tut_html = tut_data.get("html", "")
                        tut_soup = BeautifulSoup(tut_html, "html.parser")
                        return tut_soup.get_text("\n", strip=True)
                except Exception as exc:
                    logger.warning("Failed to fetch dynamic tutorial for %s: %s", target_code, exc)

        problem_patterns = [
            f"{contest_id}{index}",
            f"{contest_id} {index}",
            f"Problem {index}",
            f"{index}.",
            f"{index} ",
        ]

        # Strategy 1: spoiler divs
        for spoiler in soup.find_all("div", class_="spoiler"):
            title_el = spoiler.find(class_="spoiler-title")
            if title_el:
                title_text = title_el.get_text(strip=True)
                for pattern in problem_patterns:
                    if pattern.lower() in title_text.lower():
                        content = spoiler.find("div", class_="spoiler-content")
                        if content:
                            return content.get_text("\n", strip=True)

        # Strategy 2: headers
        for header in soup.find_all(re.compile(r'^h[1-6]$')):
            header_text = header.get_text(strip=True)
            for pattern in problem_patterns:
                if pattern.lower() in header_text.lower():
                    parts = []
                    for sib in header.find_next_siblings():
                        if sib.name and re.match(r'^h[1-6]$', sib.name):
                            break
                        parts.append(sib.get_text("\n", strip=True))
                    if parts:
                        return "\n".join(parts)

        # Strategy 3: bold text section headers
        for b_tag in soup.find_all("b"):
            b_text = b_tag.get_text(strip=True)
            for pattern in problem_patterns:
                if pattern.lower() in b_text.lower():
                    parts = []
                    parent = b_tag.parent
                    if parent:
                        for sib in parent.find_next_siblings():
                            if sib.find("b"):
                                next_b = sib.find("b")
                                if next_b and re.search(
                                        r'[A-Z]\d*[.\s]',
                                        next_b.get_text(strip=True)):
                                    break
                            parts.append(sib.get_text("\n", strip=True))
                    if parts:
                        return "\n".join(parts)

        # Strategy 4: tutorial-content div
        tutorial = soup.find("div", class_="tutorial-content")
        if tutorial is not None:
            return tutorial.get_text("\n", strip=True)

        # Strategy 5: fallback
        content = soup.find("div", class_="content")
        if content:
            text = content.get_text("\n", strip=True)
            for pattern in problem_patterns:
                idx_pos = text.lower().find(pattern.lower())
                if idx_pos >= 0:
                    return text[idx_pos:idx_pos + 3000]
            return text[:3000]

        return ""

    # ------------------------------------------------------------------
    # Accepted solutions
    # ------------------------------------------------------------------
    def _scrape_accepted_solutions(self, contest_id: str, index: str,
                                   max_solutions: int = 2) -> list[str]:
        submission_ids = self._get_accepted_submission_ids_api(
            contest_id, index, max_solutions
        )
        if not submission_ids:
            submission_ids = self._get_accepted_submission_ids_scrape(
                contest_id, index, max_solutions
            )

        solutions = []
        for sid in submission_ids:
            code = self._fetch_submission_code(contest_id, sid)
            if code:
                url = f"{BASE_URL}/contest/{contest_id}/submission/{sid}"
                solutions.append(f"// Source: {url}\n{code}")
        return solutions

    def _get_accepted_submission_ids_api(self, contest_id: str, index: str,
                                         max_solutions: int) -> list[str]:
        try:
            result = self._api_get("contest.status", params={
                "contestId": contest_id,
                "from": 1,
                "count": 50,
            })
        except Exception:
            logger.debug("API contest.status failed for contest %s",
                         contest_id)
            return []

        ids = []
        seen_authors = set()
        for sub in result:
            prob = sub.get("problem", {})
            if prob.get("index") != index:
                continue
            if sub.get("verdict") != "OK":
                continue
            author = (sub.get("author", {})
                      .get("members", [{}])[0]
                      .get("handle", ""))
            if author in seen_authors:
                continue
            seen_authors.add(author)
            ids.append(str(sub["id"]))
            if len(ids) >= max_solutions:
                break
        return ids

    def _get_accepted_submission_ids_scrape(self, contest_id: str, index: str,
                                            max_solutions: int) -> list[str]:
        status_url = (
            f"{BASE_URL}/problemset/status/{contest_id}/problem/{index}"
        )
        try:
            resp = self._get(status_url)
        except requests.RequestException:
            logger.warning("Failed to fetch status page: %s", status_url)
            return []

        soup = BeautifulSoup(resp.text, "html.parser")
        table = soup.find("table", class_="status-frame-datatable")
        if table is None:
            return []

        rows = table.find_all("tr")[1:]
        submission_ids = []
        for row in rows:
            cells = row.find_all("td")
            if len(cells) < 6:
                continue
            verdict_cell = cells[5]
            if verdict_cell.get_text(strip=True) == "Accepted":
                link = cells[0].find("a")
                if link:
                    sid = link.get_text(strip=True)
                    if sid.isdigit():
                        submission_ids.append(sid)
            if len(submission_ids) >= max_solutions:
                break
        return submission_ids

    def _fetch_submission_code(self, contest_id: str,
                               submission_id: str) -> Optional[str]:
        sub_url = f"{BASE_URL}/contest/{contest_id}/submission/{submission_id}"
        self.limiter.acquire()
        try:
            resp = self.session.get(sub_url, timeout=30, allow_redirects=True)
        except requests.RequestException:
            return None

        if resp.status_code != 200:
            sub_url = f"{BASE_URL}/problemset/submission/{contest_id}/{submission_id}"
            try:
                resp = self.session.get(sub_url, timeout=30,
                                        allow_redirects=True)
            except requests.RequestException:
                return None

        if resp.status_code != 200:
            logger.warning("Cannot fetch submission %s (status %d)",
                           submission_id, resp.status_code)
            return None

        soup = BeautifulSoup(resp.text, "html.parser")

        pre = soup.find("pre", id="program-source-text")
        if pre is not None:
            return pre.get_text("\n", strip=True)

        for pre in soup.find_all("pre", class_="program-source"):
            return pre.get_text("\n", strip=True)

        source_div = soup.find("div", id="program-source-text")
        if source_div:
            return source_div.get_text("\n", strip=True)

        return None

    # ------------------------------------------------------------------
    # Full scrape for one problem
    # ------------------------------------------------------------------
    def scrape_problem(self, problem_id: str,
                       api_rating: Optional[int] = None,
                       skip_editorial: bool = False,
                       skip_solutions: bool = False) -> dict:
        contest_id, index = self.parse_problem_id(problem_id)
        url = self.problem_url(contest_id, index)

        resp = self._get(url)
        soup = BeautifulSoup(resp.text, "html.parser")

        problem_statement = self._extract_problem_statement(soup)
        time_limit = self._extract_time_limit(soup)

        difficulty = api_rating or self._get_api_rating(problem_id)
        if difficulty is None:
            difficulty = self._extract_difficulty(soup)

        editorial_text = ""
        if not skip_editorial:
            editorial_url = self._find_editorial_url(contest_id)
            if editorial_url:
                editorial_text = self._scrape_editorial_text(
                    editorial_url, contest_id, index
                )

        accepted_solutions = []
        if not skip_solutions:
            accepted_solutions = self._scrape_accepted_solutions(
                contest_id, index
            )

        return {
            "id": problem_id,
            "source": "Codeforces",
            "url": url,
            "problem_statement": problem_statement,
            "editorial_text": editorial_text,
            "accepted_solutions": accepted_solutions,
            "time_limit": time_limit,
            "difficulty_rating": difficulty,
        }

    # ------------------------------------------------------------------
    # Batch scrape (list of IDs) – sequential
    # ------------------------------------------------------------------
    def scrape_problems(self, problem_ids: list[str],
                        output_file: Optional[str] = None) -> list[dict]:
        results = []
        for i, pid in enumerate(problem_ids):
            logger.info("[%d/%d] Scraping %s ...", i + 1, len(problem_ids), pid)
            try:
                data = self.scrape_problem(pid)
                results.append(data)
                logger.info("  ✓ %s scraped successfully", pid)
            except Exception as exc:
                logger.error("  ✗ Failed to scrape %s: %s", pid, exc)
                results.append({
                    "id": pid,
                    "source": "Codeforces",
                    "error": str(exc),
                })
            if output_file is not None:
                with open(output_file, "w") as f:
                    json.dump(results, f, indent=2, default=str)
        return results

    # ------------------------------------------------------------------
    # Scrape ALL problems – PARALLEL with checkpoint/resume
    # ------------------------------------------------------------------
    @staticmethod
    def _parse_rating(rating) -> Optional[int]:
        if isinstance(rating, int):
            return rating
        if isinstance(rating, str) and rating.isdigit():
            return int(rating)
        return None

    def _scrape_one(self, pid: str, api_rating: Optional[int],
                    total: int,
                    output_file: str, checkpoint_file: str,
                    skip_editorial: bool, skip_solutions: bool) -> bool:
        """
        Scrape a single problem and write results (called from worker thread).
        Returns True on success, False on failure.
        """
        try:
            data = self.scrape_problem(
                pid,
                api_rating=api_rating,
                skip_editorial=skip_editorial,
                skip_solutions=skip_solutions,
            )
            ok = True
        except Exception as exc:
            logger.error("  ✗ %s failed: %s", pid, exc)
            data = {"id": pid, "source": "Codeforces", "error": str(exc)}
            ok = False

        # Thread-safe file write
        with self._file_lock:
            with open(output_file, "a") as f:
                f.write(json.dumps(data, default=str) + "\n")
            with open(checkpoint_file, "a") as f:
                f.write(pid + "\n")

        # Progress
        with self._progress_lock:
            if ok:
                self._done_count += 1
            else:
                self._error_count += 1
            done = self._done_count
            errs = self._error_count

        stmt_len = len(data.get("problem_statement", ""))
        edit_len = len(data.get("editorial_text", ""))
        sol_cnt = len(data.get("accepted_solutions", []))

        if ok:
            logger.info(
                "  ✓ [%d/%d] %s  (stmt=%d  edit=%d  sols=%d)  errors=%d",
                done + errs, total, pid, stmt_len, edit_len, sol_cnt, errs,
            )
        return ok

    def scrape_all(self, output_file: str, checkpoint_file: str,
                   workers: int = DEFAULT_WORKERS,
                   max_problems: Optional[int] = None,
                   skip_editorial: bool = False,
                   skip_solutions: bool = False,
                   min_rating: Optional[int] = None,
                   max_rating: Optional[int] = None):
        """
        Scrape all Codeforces problems in parallel.
        Automatically resumes from checkpoint.
        """
        problems = self.fetch_problem_list()

        # Load checkpoint
        seen = set()
        if os.path.exists(checkpoint_file):
            with open(checkpoint_file) as f:
                seen = set(line.strip() for line in f if line.strip())
            logger.info("Checkpoint: %d problems already scraped", len(seen))

        # Filter
        filtered = []
        for p in problems:
            cid = p.get("contestId")
            idx = p.get("index")
            if cid is None or not idx:
                continue

            pid = self.make_problem_id(cid, idx)
            if pid in seen:
                continue

            rating = self._parse_rating(p.get("rating"))
            if min_rating is not None and (rating is None
                                           or rating < min_rating):
                continue
            if max_rating is not None and (rating is not None
                                           and rating > max_rating):
                continue

            filtered.append((pid, rating, str(cid)))

        if max_problems is not None:
            filtered = filtered[:max_problems]

        total = len(filtered)
        logger.info(
            "Problems to scrape: %d  (skipping %d already done)  "
            "workers=%d  rps=%.1f",
            total, len(seen), workers, self.limiter.rate,
        )

        if not filtered:
            logger.info("Nothing to scrape – all done!")
            return

        os.makedirs(os.path.dirname(output_file) or ".", exist_ok=True)

        # Sort by contest ID so threads working on the same contest
        # share the editorial URL cache efficiently
        filtered.sort(key=lambda x: x[2])

        # Reset counters
        self._done_count = 0
        self._error_count = 0

        t0 = time.monotonic()

        with ThreadPoolExecutor(max_workers=workers,
                                thread_name_prefix="cf-worker") as pool:
            futures = {}
            for pid, api_rating, _cid in filtered:
                fut = pool.submit(
                    self._scrape_one,
                    pid, api_rating, total,
                    output_file, checkpoint_file,
                    skip_editorial, skip_solutions,
                )
                futures[fut] = pid

            # Wait for all futures and log any unexpected exceptions
            for fut in as_completed(futures):
                pid = futures[fut]
                try:
                    fut.result()
                except Exception as exc:
                    logger.error("Unexpected error in worker for %s: %s",
                                 pid, exc)

        elapsed = time.monotonic() - t0
        logger.info(
            "\n" + "=" * 60 + "\n"
            "Scraping complete in %.1f seconds (%.1f problems/sec)\n"
            "  Succeeded: %d\n"
            "  Failed:    %d\n"
            "  Total:     %d\n"
            "  Output:    %s\n"
            "  Checkpoint:%s\n" + "=" * 60,
            elapsed, total / max(elapsed, 0.01),
            self._done_count, self._error_count, total,
            output_file, checkpoint_file,
        )

    # ------------------------------------------------------------------
    # Convert JSONL → JSON array
    # ------------------------------------------------------------------
    @staticmethod
    def jsonl_to_json(jsonl_path: str, json_path: str):
        records = []
        with open(jsonl_path) as f:
            for line in f:
                line = line.strip()
                if line:
                    records.append(json.loads(line))
        with open(json_path, "w") as f:
            json.dump(records, f, indent=2, default=str)
        logger.info("Converted %d records: %s → %s",
                     len(records), jsonl_path, json_path)


# ======================================================================
# CLI
# ======================================================================
def main():
    import argparse

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] [%(threadName)s] %(message)s",
    )

    parser = argparse.ArgumentParser(
        description="Scrape Codeforces problems into JSON (parallel).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s --all                          Scrape everything (4 workers)
  %(prog)s --all --workers 8              Scrape with 8 parallel workers
  %(prog)s --all --max 20                 Scrape 20 problems (for testing)
  %(prog)s --all --min-rating 1200        Only problems rated ≥ 1200
  %(prog)s --all --light --workers 12     Fast mode, 12 workers
  %(prog)s CF-1560A CF-4B                 Scrape specific problems
  %(prog)s --convert                      Convert JSONL → JSON array
        """,
    )
    parser.add_argument(
        "problems", nargs="*",
        help="Problem IDs, e.g. CF-1560A CF-4B",
    )
    parser.add_argument(
        "-o", "--output",
        help="Output file path (default: data/codeforces_problems.jsonl)",
    )
    parser.add_argument(
        "--workers", "-w", type=int, default=DEFAULT_WORKERS,
        help=f"Number of parallel workers (default: {DEFAULT_WORKERS})",
    )
    parser.add_argument(
        "--rps", type=float, default=DEFAULT_RPS,
        help=f"Max requests per second across all workers (default: {DEFAULT_RPS})",
    )
    parser.add_argument(
        "--cookies", type=str,
        help="JSON file with Codeforces session cookies",
    )
    parser.add_argument(
        "--all", action="store_true",
        help="Scrape ALL problems from the Codeforces problemset",
    )
    parser.add_argument(
        "--checkpoint", type=str,
        default=str(DEFAULT_CHECKPOINT),
        help=f"Checkpoint file for --all (default: {DEFAULT_CHECKPOINT})",
    )
    parser.add_argument(
        "--max", type=int, default=None,
        help="Max problems to scrape (for --all, useful for testing)",
    )
    parser.add_argument(
        "--light", action="store_true",
        help="Skip editorial & solutions (much faster)",
    )
    parser.add_argument(
        "--min-rating", type=int, default=None,
        help="Minimum difficulty rating filter",
    )
    parser.add_argument(
        "--max-rating", type=int, default=None,
        help="Maximum difficulty rating filter",
    )
    parser.add_argument(
        "--convert", action="store_true",
        help="Convert JSONL output file to a single JSON array",
    )
    args = parser.parse_args()

    if not args.output:
        args.output = str(DEFAULT_OUTPUT)

    if args.convert:
        jsonl_path = args.output
        json_path = jsonl_path.replace(".jsonl", ".json")
        if json_path == jsonl_path:
            json_path += ".json"
        CodeforcesScraper.jsonl_to_json(jsonl_path, json_path)
        return

    DATA_DIR.mkdir(parents=True, exist_ok=True)

    cookies = {}
    if args.cookies:
        with open(args.cookies) as f:
            cookies = json.load(f)

    scraper = CodeforcesScraper(rps=args.rps, cookies=cookies)

    if args.all:
        scraper.scrape_all(
            output_file=args.output,
            checkpoint_file=args.checkpoint,
            workers=args.workers,
            max_problems=args.max,
            skip_editorial=args.light,
            skip_solutions=args.light,
            min_rating=args.min_rating,
            max_rating=args.max_rating,
        )
    elif args.problems:
        results = scraper.scrape_problems(
            args.problems, output_file=args.output,
        )
        if args.output == str(DEFAULT_OUTPUT):
            print(json.dumps(results, indent=2, default=str))
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
