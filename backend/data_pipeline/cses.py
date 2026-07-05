import json
import logging
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import cloudscraper
from bs4 import BeautifulSoup

logger = logging.getLogger(__name__)

BASE_URL = "https://cses.fi"
DEFAULT_WORKERS = 4
DEFAULT_RPS = 4.0

DATA_DIR = Path(__file__).resolve().parent / "data"
DEFAULT_OUTPUT = DATA_DIR / "cses_solutions.jsonl"
DEFAULT_CHECKPOINT = DATA_DIR / "cses_checkpoint.txt"
SOLUTIONS_DIR = DATA_DIR / "CSES-Solutions"


class RateLimiter:
    """Thread-safe token-bucket rate limiter."""

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
                self.tokens = min(self.capacity, self.tokens + elapsed * self.rate)
                self.last = now
                if self.tokens >= 1.0:
                    self.tokens -= 1.0
                    return
            time.sleep(1.0 / self.rate)


class CSESScraper:
    def __init__(self, rps: float = DEFAULT_RPS):
        # cloudscraper mimics a real browser to bypass potential Cloudflare protections
        self.session = cloudscraper.create_scraper(
            browser={"browser": "chrome", "platform": "windows", "desktop": True}
        )
        self.limiter = RateLimiter(rps)
        self._file_lock = threading.Lock()
        self._progress_lock = threading.Lock()
        self._done_count = 0
        self._error_count = 0
        
        # Build mapping of problem title to local solution file path
        self._solution_files = {}
        if SOLUTIONS_DIR.exists():
            for root, _, files in os.walk(SOLUTIONS_DIR):
                for file in files:
                    if file.endswith(".cpp"):
                        title = file[:-4]  # Remove .cpp
                        self._solution_files[title] = Path(root) / file

    def _get(self, url: str, retries: int = 3):
        self.limiter.acquire()
        for attempt in range(retries):
            try:
                logger.debug("GET %s (attempt %d)", url, attempt + 1)
                resp = self.session.get(url, timeout=30)
                if resp.status_code == 429:
                    wait = 2.0 ** (attempt + 1)
                    logger.warning("Rate limited on %s, waiting %.1fs", url, wait)
                    time.sleep(wait)
                    self.limiter.acquire()
                    continue
                resp.raise_for_status()
                return resp
            except Exception as exc:
                if attempt == retries - 1:
                    raise
                wait = 2.0 ** (attempt + 1)
                logger.warning("Request failed (%s), retry in %.1fs", exc, wait)
                time.sleep(wait)
                self.limiter.acquire()
        raise RuntimeError(f"Exhausted retries for {url}")

    def fetch_problem_list(self) -> list[dict]:
        """Fetch all problems from the CSES list page."""
        logger.info("Fetching problem list from CSES...")
        url = f"{BASE_URL}/problemset/list/"
        resp = self._get(url)
        soup = BeautifulSoup(resp.text, "html.parser")

        problems = []
        # The problems are usually listed under <h2> category headers
        # Inside ul.task-list -> li.task -> a

        for h2 in soup.find_all("h2"):
            category = h2.get_text(strip=True)
            ul = h2.find_next_sibling("ul", class_="task-list")
            if not ul:
                continue

            for a in ul.find_all("a", href=True):
                href = a["href"]
                if "/problemset/task/" in href:
                    task_id = href.strip("/").split("/")[-1]
                    title = a.get_text(strip=True)
                    problems.append(
                        {
                            "id": f"CSES-{task_id}",
                            "task_id": task_id,
                            "title": title,
                            "category": category,
                            "url": f"{BASE_URL}{href}",
                        }
                    )

        logger.info(
            "Found %d problems across %d categories",
            len(problems),
            len(set(p["category"] for p in problems)),
        )
        return problems

    def scrape_problem(self, problem_meta: dict) -> dict:
        """Scrape a single CSES problem page."""
        url = problem_meta["url"]
        resp = self._get(url)
        soup = BeautifulSoup(resp.text, "html.parser")

        content_div = soup.find("div", class_="content")
        if not content_div:
            raise ValueError(f"Could not find content div for {url}")

        # Extract time and memory limits
        text = content_div.get_text("\n", strip=True)

        # CSES puts limits at the top of the content div in unordered lists
        time_limit = ""
        memory_limit = ""

        for li in content_div.find_all("li"):
            li_text = li.get_text(strip=True)
            if "Time limit:" in li_text:
                time_limit = li_text.replace("Time limit:", "").strip()
            elif "Memory limit:" in li_text:
                memory_limit = li_text.replace("Memory limit:", "").strip()

        # Remove the limits from the statement text for cleaner output
        # Sometimes they are in an info box or list at the top
        ul_info = content_div.find("ul", class_="task-constraints")
        if ul_info:
            ul_info.decompose()  # Remove it from the tree before getting text again
            text = content_div.get_text("\n", strip=True)

        # Check if we have a local solution file
        accepted_solutions = []
        if problem_meta["title"] in self._solution_files:
            sol_path = self._solution_files[problem_meta["title"]]
            try:
                with open(sol_path, "r", encoding="utf-8") as f:
                    code = f.read()
                accepted_solutions.append(f"// Source: {sol_path.name}\n{code}")
            except Exception as e:
                logger.warning("Failed to read solution file for %s: %s", problem_meta["title"], e)

        return {
            "id": problem_meta["id"],
            "source": "CSES",
            "url": url,
            "title": problem_meta["title"],
            "category": problem_meta["category"],
            "problem_statement": text,
            "time_limit": time_limit,
            "memory_limit": memory_limit,
            "accepted_solutions": accepted_solutions,
        }

    def _scrape_one(
        self, problem_meta: dict, total: int, output_file: str, checkpoint_file: str
    ) -> bool:
        pid = problem_meta["id"]
        try:
            data = self.scrape_problem(problem_meta)
            ok = True
        except Exception as exc:
            logger.error("  ✗ %s failed: %s", pid, exc)
            data = {"id": pid, "source": "CSES", "error": str(exc)}
            ok = False

        # Thread-safe file write
        with self._file_lock:
            with open(output_file, "a", encoding="utf-8") as f:
                f.write(json.dumps(data, default=str) + "\n")
            with open(checkpoint_file, "a", encoding="utf-8") as f:
                f.write(pid + "\n")

        with self._progress_lock:
            if ok:
                self._done_count += 1
            else:
                self._error_count += 1
            done = self._done_count
            errs = self._error_count

        if ok:
            stmt_len = len(data.get("problem_statement", ""))
            logger.info(
                "  ✓ [%d/%d] %s: %s (stmt=%d chars) errors=%d",
                done + errs,
                total,
                pid,
                problem_meta["title"],
                stmt_len,
                errs,
            )
        return ok

    def scrape_all(
        self, output_file: str, checkpoint_file: str, workers: int = DEFAULT_WORKERS
    ):
        problems = self.fetch_problem_list()

        seen = set()
        if os.path.exists(checkpoint_file):
            with open(checkpoint_file, "r", encoding="utf-8") as f:
                seen = set(line.strip() for line in f if line.strip())
            logger.info("Checkpoint: %d problems already scraped", len(seen))

        filtered = [p for p in problems if p["id"] not in seen]
        total = len(filtered)

        if not filtered:
            logger.info("Nothing to scrape – all done!")
            return

        os.makedirs(os.path.dirname(output_file) or ".", exist_ok=True)

        logger.info("Starting scrape of %d problems using %d workers", total, workers)

        t0 = time.monotonic()

        with ThreadPoolExecutor(
            max_workers=workers, thread_name_prefix="cses-worker"
        ) as pool:
            futures = {
                pool.submit(
                    self._scrape_one, p, total, output_file, checkpoint_file
                ): p["id"]
                for p in filtered
            }

            for fut in as_completed(futures):
                pid = futures[fut]
                try:
                    fut.result()
                except Exception as exc:
                    logger.error("Unexpected error in worker for %s: %s", pid, exc)

        elapsed = time.monotonic() - t0
        logger.info(
            "\n" + "=" * 60 + "\n"
            "Scraping complete in %.1f seconds (%.1f problems/sec)\n"
            "  Succeeded: %d\n"
            "  Failed:    %d\n"
            "  Total:     %d\n"
            "  Output:    %s\n"
            "  Checkpoint:%s\n" + "=" * 60,
            elapsed,
            total / max(elapsed, 0.01),
            self._done_count,
            self._error_count,
            total,
            output_file,
            checkpoint_file,
        )


def main():
    import argparse

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] [%(threadName)s] %(message)s",
    )

    parser = argparse.ArgumentParser(description="Scrape CSES Problem Set into JSONL.")
    parser.add_argument(
        "-o", "--output", default=str(DEFAULT_OUTPUT), help="Output JSONL file path"
    )
    parser.add_argument(
        "--workers",
        "-w",
        type=int,
        default=DEFAULT_WORKERS,
        help="Number of parallel workers",
    )
    parser.add_argument(
        "--rps", type=float, default=DEFAULT_RPS, help="Max requests per second"
    )
    parser.add_argument(
        "--checkpoint", default=str(DEFAULT_CHECKPOINT), help="Checkpoint file path"
    )
    args = parser.parse_args()

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    scraper = CSESScraper(rps=args.rps)
    scraper.scrape_all(
        output_file=args.output, checkpoint_file=args.checkpoint, workers=args.workers
    )


if __name__ == "__main__":
    main()
