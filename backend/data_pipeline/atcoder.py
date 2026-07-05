"""
AtCoder Scraper – parallel scraping of problems with statements,
editorials, and accepted solutions.

Usage:
    # Scrape everything with 8 parallel workers:
    python atcoder.py --all --workers 8

    # Scrape with filters:
    python atcoder.py --all --min-rating 100 --max-rating 400

    # Scrape specific problems:
    python atcoder.py AC-abc300_a AC-abc001_1

    # Light mode (no editorial/solutions, much faster):
    python atcoder.py --all --light --workers 12

    # Test with a small batch:
    python atcoder.py --all --max 10

Output is written to data/atcoder_problems.jsonl (one JSON object per line).
"""

import json
import logging
import os
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Optional

import cloudscraper
import requests
from bs4 import BeautifulSoup

logger = logging.getLogger(__name__)

BASE_URL = "https://atcoder.jp"
MERGED_PROBLEMS_URL = (
    "https://kenkoooo.com/atcoder/resources/merged-problems.json"
)
MAX_RETRIES = 3
RETRY_BACKOFF = 2.0

DEFAULT_WORKERS = 4
DEFAULT_RPS = 2.0

DATA_DIR = Path(__file__).resolve().parent / "data"
DEFAULT_OUTPUT = DATA_DIR / "atcoder_problems.jsonl"
DEFAULT_CHECKPOINT = DATA_DIR / "atcoder_checkpoint.txt"


class RateLimiter:
    """Token-bucket rate limiter shared across all threads."""

    def __init__(self, rate: float):
        self.rate = rate
        self.capacity = max(rate, 1.0)
        self.tokens = self.capacity
        self.last = time.monotonic()
        self._lock = threading.Lock()

    def acquire(self):
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
            time.sleep(1.0 / self.rate)


class AtCoderScraper:
    def __init__(self, rps: float = DEFAULT_RPS,
                 cookies: Optional[dict] = None):
        browser_kwargs = {
            "browser": "chrome",
            "platform": "windows",
            "desktop": True,
        }

        custom_ua = None
        if cookies and "User-Agent" in cookies:
            custom_ua = cookies.pop("User-Agent")

        self.session = cloudscraper.create_scraper(browser=browser_kwargs)
        if custom_ua:
            self.session.headers.update({"User-Agent": custom_ua})
        if cookies:
            self.session.cookies.update(cookies)

        self.limiter = RateLimiter(rps)
        self._merged_problems: dict[str, dict] = {}
        self._file_lock = threading.Lock()
        self._progress_lock = threading.Lock()
        self._done_count = 0
        self._error_count = 0

    def set_cookies(self, cookies: dict):
        self.session.cookies.update(cookies)

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

    @staticmethod
    def parse_problem_id(problem_id: str) -> str:
        """Parse 'AC-abc300_a' → 'abc300_a'."""
        match = re.fullmatch(r"AC-([a-zA-Z0-9_]+)", problem_id)
        if not match:
            raise ValueError(
                f"Invalid problem ID '{problem_id}'. Expected AC-<problem_id> "
                f"(e.g. AC-abc300_a, AC-abc001_1)."
            )
        return match.group(1)

    @staticmethod
    def make_problem_id(atcoder_id: str) -> str:
        return f"AC-{atcoder_id}"

    @staticmethod
    def contest_from_problem_id(atcoder_id: str) -> str:
        return atcoder_id.rsplit("_", 1)[0]

    @staticmethod
    def problem_url(contest_id: str, atcoder_id: str) -> str:
        return f"{BASE_URL}/contests/{contest_id}/tasks/{atcoder_id}"

    @staticmethod
    def _parse_rating(rating) -> Optional[int]:
        if rating is None:
            return None
        if isinstance(rating, int):
            return rating
        if isinstance(rating, float):
            return int(rating)
        if isinstance(rating, str) and rating.replace(".", "", 1).isdigit():
            return int(float(rating))
        return None

    def fetch_problem_list(self) -> list[dict]:
        """Return all problems from the kenkoooo merged-problems API."""
        logger.info("Fetching problem list from kenkoooo API ...")
        self.limiter.acquire()
        resp = self.session.get(MERGED_PROBLEMS_URL, timeout=120)
        resp.raise_for_status()
        problems = resp.json()

        self._merged_problems = {p["id"]: p for p in problems}
        logger.info("API returned %d problems", len(problems))
        return problems

    def _get_merged_problem(self, atcoder_id: str) -> Optional[dict]:
        if not self._merged_problems:
            self.fetch_problem_list()
        return self._merged_problems.get(atcoder_id)

    def _contest_id_for_problem(self, atcoder_id: str,
                                merged: Optional[dict] = None) -> str:
        if merged:
            for key in ("shortest_contest_id", "fastest_contest_id",
                        "first_contest_id", "contest_id"):
                cid = merged.get(key)
                if cid and not cid.startswith("adt_"):
                    return cid
        return self.contest_from_problem_id(atcoder_id)

    def _extract_problem_statement(self, soup: BeautifulSoup) -> str:
        stmt = soup.find("div", id="task-statement")
        if stmt is None:
            return ""

        lang_en = stmt.find("span", class_="lang-en")
        if lang_en is not None:
            return lang_en.get_text("\n", strip=True)
        return stmt.get_text("\n", strip=True)

    def _extract_time_limit(self, soup: BeautifulSoup) -> str:
        for node in soup.find_all(string=re.compile(r"Time Limit\s*:", re.I)):
            text = node if isinstance(node, str) else node.get_text(" ", strip=True)
            match = re.search(
                r"Time Limit\s*:\s*(.+?)(?:\s*/\s*Memory Limit|$)",
                text,
                re.I,
            )
            if match:
                return match.group(1).strip()
        return ""

    def _find_editorial_urls(self, contest_id: str,
                             atcoder_id: str) -> list[str]:
        editorial_list_url = (
            f"{BASE_URL}/contests/{contest_id}/tasks/{atcoder_id}/editorial"
        )
        try:
            resp = self._get(editorial_list_url)
        except requests.RequestException:
            logger.warning("Failed to fetch editorial list: %s",
                           editorial_list_url)
            return []

        soup = BeautifulSoup(resp.text, "html.parser")
        main = soup.find("div", id="main-container")
        if main is None:
            return []

        urls = []
        seen = set()
        for a in main.find_all("a", href=True):
            href = a["href"]
            if "/editorial/" not in href:
                continue
            text = a.get_text(strip=True)
            if text in {"Editorial", "Another Solution",
                        "Proposer's Implementation"}:
                full = href if href.startswith("http") else BASE_URL + href
                if full not in seen:
                    seen.add(full)
                    urls.append(full)
        return urls

    def _scrape_editorial_page(self, editorial_url: str) -> tuple[str, list[str]]:
        try:
            resp = self._get(editorial_url)
        except requests.RequestException:
            logger.warning("Failed to fetch editorial at %s", editorial_url)
            return "", []

        soup = BeautifulSoup(resp.text, "html.parser")
        main = soup.find("div", id="main-container")
        if main is None:
            return "", []

        content_div = None
        h2 = main.find("h2")
        if h2 is not None:
            parent = h2.find_parent("div", class_=lambda c: c and "col-sm-12" in c)
            if parent is not None:
                content_div = parent

        container = content_div or main
        codes = []
        for pre in container.find_all("pre"):
            code = pre.get_text("\n", strip=True)
            if code:
                codes.append(code)
            pre.decompose()

        text = container.get_text("\n", strip=True)
        text = re.sub(r"\n{3,}", "\n\n", text).strip()
        return text, codes

    def _scrape_editorial_text(self, contest_id: str,
                               atcoder_id: str) -> str:
        editorial_urls = self._find_editorial_urls(contest_id, atcoder_id)
        if not editorial_urls:
            return ""

        parts = []
        for url in editorial_urls[:3]:
            text, _codes = self._scrape_editorial_page(url)
            if text:
                parts.append(text)
        return "\n\n".join(parts)

    def _submission_candidates(self, merged: Optional[dict]) -> list[tuple[str, str]]:
        if not merged:
            return []

        candidates = []
        seen = set()
        for prefix in ("fastest", "first", "shortest"):
            sid = merged.get(f"{prefix}_submission_id")
            cid = merged.get(f"{prefix}_contest_id")
            if not sid or not cid:
                continue
            key = (str(cid), str(sid))
            if key in seen:
                continue
            seen.add(key)
            candidates.append(key)
        return candidates

    def _fetch_submission_code(self, contest_id: str,
                               submission_id: str) -> Optional[str]:
        sub_url = f"{BASE_URL}/contests/{contest_id}/submissions/{submission_id}"
        self.limiter.acquire()
        try:
            resp = self.session.get(sub_url, timeout=30, allow_redirects=True)
        except requests.RequestException:
            return None

        if resp.status_code != 200:
            logger.warning("Cannot fetch submission %s (status %d)",
                           submission_id, resp.status_code)
            return None

        soup = BeautifulSoup(resp.text, "html.parser")
        pre = soup.find("pre", id="submission-code")
        if pre is not None:
            return pre.get_text("\n", strip=True)
        return None

    def _scrape_accepted_solutions(self, merged: Optional[dict],
                                   max_solutions: int = 2) -> list[str]:
        solutions = []
        for contest_id, submission_id in self._submission_candidates(merged):
            code = self._fetch_submission_code(contest_id, submission_id)
            if not code or len(code) < 20:
                continue
            url = f"{BASE_URL}/contests/{contest_id}/submissions/{submission_id}"
            solutions.append(f"// Source: {url}\n{code}")
            if len(solutions) >= max_solutions:
                break
        return solutions

    def scrape_problem(self, problem_id: str,
                       merged: Optional[dict] = None,
                       skip_editorial: bool = False,
                       skip_solutions: bool = False) -> dict:
        atcoder_id = self.parse_problem_id(problem_id)
        merged = merged or self._get_merged_problem(atcoder_id)
        contest_id = self._contest_id_for_problem(atcoder_id, merged)
        url = self.problem_url(contest_id, atcoder_id)

        resp = self._get(url)
        soup = BeautifulSoup(resp.text, "html.parser")

        problem_statement = self._extract_problem_statement(soup)
        time_limit = self._extract_time_limit(soup)
        difficulty = None
        if merged:
            difficulty = self._parse_rating(merged.get("point"))

        editorial_text = ""
        if not skip_editorial:
            editorial_text = self._scrape_editorial_text(contest_id, atcoder_id)

        accepted_solutions = []
        if not skip_solutions:
            accepted_solutions = self._scrape_accepted_solutions(merged)

        return {
            "id": problem_id,
            "source": "AtCoder",
            "url": url,
            "problem_statement": problem_statement,
            "editorial_text": editorial_text,
            "accepted_solutions": accepted_solutions,
            "time_limit": time_limit,
            "difficulty_rating": difficulty,
        }

    def scrape_problems(self, problem_ids: list[str],
                        output_file: Optional[str] = None) -> list[dict]:
        if not self._merged_problems:
            self.fetch_problem_list()

        results = []
        for i, pid in enumerate(problem_ids):
            logger.info("[%d/%d] Scraping %s ...", i + 1, len(problem_ids), pid)
            try:
                atcoder_id = self.parse_problem_id(pid)
                merged = self._get_merged_problem(atcoder_id)
                data = self.scrape_problem(pid, merged=merged)
                results.append(data)
                logger.info("  ✓ %s scraped successfully", pid)
            except Exception as exc:
                logger.error("  ✗ Failed to scrape %s: %s", pid, exc)
                results.append({
                    "id": pid,
                    "source": "AtCoder",
                    "error": str(exc),
                })
            if output_file is not None:
                with open(output_file, "w") as f:
                    json.dump(results, f, indent=2, default=str)
        return results

    def _scrape_one(self, pid: str, merged: Optional[dict], total: int,
                    output_file: str, checkpoint_file: str,
                    skip_editorial: bool, skip_solutions: bool) -> bool:
        try:
            data = self.scrape_problem(
                pid,
                merged=merged,
                skip_editorial=skip_editorial,
                skip_solutions=skip_solutions,
            )
            ok = True
        except Exception as exc:
            logger.error("  ✗ %s failed: %s", pid, exc)
            data = {"id": pid, "source": "AtCoder", "error": str(exc)}
            ok = False

        with self._file_lock:
            with open(output_file, "a") as f:
                f.write(json.dumps(data, default=str) + "\n")
            with open(checkpoint_file, "a") as f:
                f.write(pid + "\n")

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
        """Scrape all AtCoder problems in parallel with checkpoint/resume."""
        problems = self.fetch_problem_list()

        seen = set()
        if os.path.exists(checkpoint_file):
            with open(checkpoint_file) as f:
                seen = set(line.strip() for line in f if line.strip())
            logger.info("Checkpoint: %d problems already scraped", len(seen))

        filtered = []
        for p in problems:
            atcoder_id = p.get("id")
            if not atcoder_id:
                continue

            pid = self.make_problem_id(atcoder_id)
            if pid in seen:
                continue

            rating = self._parse_rating(p.get("point"))
            if min_rating is not None and (rating is None or rating < min_rating):
                continue
            if max_rating is not None and (rating is not None and rating > max_rating):
                continue

            filtered.append((pid, p))

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

        filtered.sort(key=lambda x: x[0])

        self._done_count = 0
        self._error_count = 0
        t0 = time.monotonic()

        with ThreadPoolExecutor(max_workers=workers,
                                thread_name_prefix="ac-worker") as pool:
            futures = {}
            for pid, merged in filtered:
                fut = pool.submit(
                    self._scrape_one,
                    pid, merged, total,
                    output_file, checkpoint_file,
                    skip_editorial, skip_solutions,
                )
                futures[fut] = pid

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


def main():
    import argparse

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] [%(threadName)s] %(message)s",
    )

    parser = argparse.ArgumentParser(
        description="Scrape AtCoder problems into JSONL (parallel).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s --all                          Scrape everything (4 workers)
  %(prog)s --all --workers 8              Scrape with 8 parallel workers
  %(prog)s --all --max 20                 Scrape 20 problems (for testing)
  %(prog)s --all --min-rating 300         Only problems with point >= 300
  %(prog)s --all --light --workers 12     Fast mode, 12 workers
  %(prog)s AC-abc300_a AC-abc001_1        Scrape specific problems
  %(prog)s --convert                      Convert JSONL → JSON array
        """,
    )
    parser.add_argument(
        "problems", nargs="*",
        help="Problem IDs, e.g. AC-abc300_a AC-abc001_1",
    )
    parser.add_argument(
        "-o", "--output",
        help="Output file path (default: data/atcoder_problems.jsonl)",
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
        help="JSON file with AtCoder session cookies",
    )
    parser.add_argument(
        "--all", action="store_true",
        help="Scrape ALL problems from the kenkoooo merged-problems API",
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
        help="Minimum problem point filter (AtCoder point value)",
    )
    parser.add_argument(
        "--max-rating", type=int, default=None,
        help="Maximum problem point filter (AtCoder point value)",
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
        AtCoderScraper.jsonl_to_json(jsonl_path, json_path)
        return

    DATA_DIR.mkdir(parents=True, exist_ok=True)

    cookies = {}
    if args.cookies:
        with open(args.cookies) as f:
            cookies = json.load(f)

    scraper = AtCoderScraper(rps=args.rps, cookies=cookies)

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
