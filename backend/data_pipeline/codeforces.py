"""
Codeforces Scraper – scrapes ALL problems with statements, editorials, and solutions.

Usage:
    # Scrape everything (auto-resumes from checkpoint):
    python codeforces.py --all

    # Scrape with filters:
    python codeforces.py --all --min-rating 800 --max-rating 1600

    # Scrape specific problems:
    python codeforces.py CF-1560A CF-4B

    # Light mode (no editorial/solutions, much faster):
    python codeforces.py --all --light

    # Test with a small batch:
    python codeforces.py --all --max 10

Output is written to data/codeforces_problems.jsonl (one JSON object per line).
"""

import json
import os
import requests
from bs4 import BeautifulSoup
import time
import re
import logging
from typing import Optional
from pathlib import Path

logger = logging.getLogger(__name__)

BASE_URL = "https://codeforces.com"
API_URL = f"{BASE_URL}/api"
REQUEST_DELAY = 1.0          # be polite – 1 req/sec
MAX_RETRIES = 3
RETRY_BACKOFF = 2.0          # exponential backoff factor

DATA_DIR = Path(__file__).resolve().parent / "data"
DEFAULT_OUTPUT = DATA_DIR / "codeforces_problems.jsonl"
DEFAULT_CHECKPOINT = DATA_DIR / "codeforces_checkpoint.txt"


class CodeforcesScraper:
    def __init__(self, delay: float = REQUEST_DELAY,
                 cookies: Optional[dict] = None):
        self.session = requests.Session()
        self.session.headers.update({
            "User-Agent": (
                "Mozilla/5.0 (X11; Linux x86_64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/126.0.0.0 Safari/537.36"
            )
        })
        if cookies:
            self.session.cookies.update(cookies)
        self.delay = delay
        # Cache: contest_id -> editorial_url
        self._editorial_url_cache: dict[str, Optional[str]] = {}
        # Cache: API problems for rating lookup
        self._api_ratings: dict[str, Optional[int]] = {}

    def set_cookies(self, cookies: dict):
        self.session.cookies.update(cookies)

    # ------------------------------------------------------------------
    # HTTP helpers with retry
    # ------------------------------------------------------------------
    def _wait(self):
        time.sleep(self.delay)

    def _get(self, url: str, retries: int = MAX_RETRIES) -> requests.Response:
        self._wait()
        for attempt in range(retries):
            try:
                logger.debug("GET %s (attempt %d)", url, attempt + 1)
                resp = self.session.get(url, timeout=30)
                if resp.status_code == 429:
                    wait = RETRY_BACKOFF ** (attempt + 1)
                    logger.warning("Rate limited, waiting %.1fs ...", wait)
                    time.sleep(wait)
                    continue
                resp.raise_for_status()
                return resp
            except requests.RequestException as exc:
                if attempt == retries - 1:
                    raise
                wait = RETRY_BACKOFF ** (attempt + 1)
                logger.warning(
                    "Request failed (%s), retry in %.1fs ...", exc, wait
                )
                time.sleep(wait)
        raise RuntimeError(f"Exhausted retries for {url}")   # unreachable

    def _api_get(self, endpoint: str, params: dict | None = None) -> dict:
        """Call the Codeforces API (with retry)."""
        url = f"{API_URL}/{endpoint}"
        self._wait()
        for attempt in range(MAX_RETRIES):
            try:
                resp = self.session.get(url, params=params, timeout=60)
                if resp.status_code == 429:
                    wait = RETRY_BACKOFF ** (attempt + 1)
                    logger.warning("API rate limited, waiting %.1fs ...", wait)
                    time.sleep(wait)
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
        stats = result.get("problemStatistics", [])

        # Build rating cache
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
            # Remove the label prefix like "time limit per test"
            text = re.sub(r'^time limit per test\s*', '', text, flags=re.I)
            return text.strip() or div.get_text(" ", strip=True)
        return ""

    def _extract_difficulty(self, soup: BeautifulSoup) -> Optional[int]:
        """Fallback: try to get difficulty from the problem page tags."""
        for tag in soup.find_all("span", class_="tag-box"):
            title = tag.get("title", "")
            if "difficulty" in title.lower():
                text = tag.get_text(strip=True)
                nums = re.findall(r'\d+', text)
                if nums:
                    return int(nums[0])
        return None

    # ------------------------------------------------------------------
    # Editorial: discover URL from the contest page
    # ------------------------------------------------------------------
    def _find_editorial_url(self, contest_id: str) -> Optional[str]:
        """Find the editorial blog entry URL for a contest."""
        if contest_id in self._editorial_url_cache:
            return self._editorial_url_cache[contest_id]

        # Strategy 1: Check the contest page sidebar
        contest_url = f"{BASE_URL}/contest/{contest_id}"
        try:
            resp = self._get(contest_url)
            soup = BeautifulSoup(resp.text, "html.parser")

            # Look for "Tutorial" / "Editorial" links in the sidebar
            for a in soup.find_all("a", href=True):
                txt = a.get_text(strip=True).lower()
                href = a["href"]
                if re.search(r'\btutorial\b|\beditorial\b', txt):
                    if href.startswith("/"):
                        href = BASE_URL + href
                    self._editorial_url_cache[contest_id] = href
                    logger.debug("Editorial for contest %s: %s",
                                 contest_id, href)
                    return href

            # Look for any blog/entry links on the contest page
            for a in soup.find_all("a", href=True):
                href = a["href"]
                if "/blog/entry/" in href:
                    txt = a.get_text(strip=True).lower()
                    if re.search(r'tutorial|editorial|разбор', txt):
                        full = href if href.startswith("http") else BASE_URL + href
                        self._editorial_url_cache[contest_id] = full
                        return full
        except requests.RequestException:
            logger.warning("Failed to fetch contest page %s", contest_url)

        self._editorial_url_cache[contest_id] = None
        return None

    # ------------------------------------------------------------------
    # Editorial text extraction from the blog page
    # ------------------------------------------------------------------
    def _scrape_editorial_text(self, editorial_url: str,
                               contest_id: str, index: str) -> str:
        """Scrape the editorial blog and extract text for a specific problem."""
        try:
            resp = self._get(editorial_url)
        except requests.RequestException:
            logger.warning("Failed to fetch editorial at %s", editorial_url)
            return ""

        soup = BeautifulSoup(resp.text, "html.parser")

        # The problem identifier in editorials (e.g. "1560A", "A", "Problem A")
        problem_patterns = [
            f"{contest_id}{index}",           # "1560A"
            f"{contest_id} {index}",          # "1560 A"
            f"Problem {index}",               # "Problem A"
            f"{index}.",                       # "A."
            f"{index} ",                       # "A " at start
        ]

        # Strategy 1: Look for tutorial-content divs (Codeforces blog spoiler format)
        for spoiler in soup.find_all("div", class_="spoiler"):
            title_el = spoiler.find(class_="spoiler-title")
            if title_el:
                title_text = title_el.get_text(strip=True)
                for pattern in problem_patterns:
                    if pattern.lower() in title_text.lower():
                        content = spoiler.find("div", class_="spoiler-content")
                        if content:
                            return content.get_text("\n", strip=True)

        # Strategy 2: Look for headers matching the problem
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

        # Strategy 3: Bold text section headers
        for b_tag in soup.find_all("b"):
            b_text = b_tag.get_text(strip=True)
            for pattern in problem_patterns:
                if pattern.lower() in b_text.lower():
                    parts = []
                    parent = b_tag.parent
                    if parent:
                        for sib in parent.find_next_siblings():
                            if sib.find("b"):
                                # Check if this bold starts a new problem section
                                next_b = sib.find("b")
                                if next_b and re.search(
                                        r'[A-Z]\d*[.\s]', next_b.get_text(strip=True)):
                                    break
                            parts.append(sib.get_text("\n", strip=True))
                    if parts:
                        return "\n".join(parts)

        # Strategy 4: tutorial-content div (older format)
        tutorial = soup.find("div", class_="tutorial-content")
        if tutorial is not None:
            return tutorial.get_text("\n", strip=True)

        # Strategy 5: fallback – grab the whole blog content
        content = soup.find("div", class_="content")
        if content:
            text = content.get_text("\n", strip=True)
            # Try to find the relevant section
            for pattern in problem_patterns:
                idx_pos = text.lower().find(pattern.lower())
                if idx_pos >= 0:
                    # Extract ~3000 chars from this point
                    return text[idx_pos:idx_pos + 3000]
            return text[:3000]

        return ""

    # ------------------------------------------------------------------
    # Accepted solutions – fetch from status page + submission source
    # ------------------------------------------------------------------
    def _scrape_accepted_solutions(self, contest_id: str, index: str,
                                   max_solutions: int = 2) -> list[str]:
        """Get source code of accepted solutions for a problem."""
        # Try the API first (more reliable than scraping the status page)
        submission_ids = self._get_accepted_submission_ids_api(
            contest_id, index, max_solutions
        )

        # Fallback to scraping the status page
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
        """Use contest.status API to find accepted submission IDs."""
        try:
            result = self._api_get("contest.status", params={
                "contestId": contest_id,
                "from": 1,
                "count": 50,  # check first 50 submissions
            })
        except Exception:
            logger.debug("API contest.status failed for contest %s", contest_id)
            return []

        ids = []
        seen_authors = set()
        for sub in result:
            prob = sub.get("problem", {})
            if prob.get("index") != index:
                continue
            if sub.get("verdict") != "OK":
                continue
            # One solution per author to get diverse solutions
            author = sub.get("author", {}).get("members", [{}])[0].get("handle", "")
            if author in seen_authors:
                continue
            seen_authors.add(author)
            ids.append(str(sub["id"]))
            if len(ids) >= max_solutions:
                break
        return ids

    def _get_accepted_submission_ids_scrape(self, contest_id: str, index: str,
                                            max_solutions: int) -> list[str]:
        """Scrape the status page to find accepted submission IDs."""
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
        """Download the source code for a specific submission."""
        # Try the contest URL format first
        sub_url = f"{BASE_URL}/contest/{contest_id}/submission/{submission_id}"
        self._wait()
        try:
            resp = self.session.get(sub_url, timeout=30, allow_redirects=True)
        except requests.RequestException:
            return None

        if resp.status_code != 200:
            # Try the problemset URL format
            sub_url = f"{BASE_URL}/problemset/submission/{contest_id}/{submission_id}"
            try:
                resp = self.session.get(sub_url, timeout=30, allow_redirects=True)
            except requests.RequestException:
                return None

        if resp.status_code != 200:
            logger.warning("Cannot fetch submission %s (status %d)",
                           submission_id, resp.status_code)
            return None

        soup = BeautifulSoup(resp.text, "html.parser")

        # Try multiple selectors for the source code
        pre = soup.find("pre", id="program-source-text")
        if pre is not None:
            return pre.get_text("\n", strip=True)

        for pre in soup.find_all("pre", class_="program-source"):
            return pre.get_text("\n", strip=True)

        # Some pages have it in a different div
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
        """Scrape a single problem: statement, editorial, solutions."""
        contest_id, index = self.parse_problem_id(problem_id)
        url = self.problem_url(contest_id, index)

        resp = self._get(url)
        soup = BeautifulSoup(resp.text, "html.parser")

        problem_statement = self._extract_problem_statement(soup)
        time_limit = self._extract_time_limit(soup)

        # Use API rating if available, fallback to page scrape
        difficulty = api_rating or self._get_api_rating(problem_id)
        if difficulty is None:
            difficulty = self._extract_difficulty(soup)

        # Editorial
        editorial_text = ""
        if not skip_editorial:
            editorial_url = self._find_editorial_url(contest_id)
            if editorial_url:
                editorial_text = self._scrape_editorial_text(
                    editorial_url, contest_id, index
                )

        # Accepted solutions
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
    # Batch scrape (list of IDs)
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
            # Save incrementally
            if output_file is not None:
                with open(output_file, "w") as f:
                    json.dump(results, f, indent=2, default=str)
        return results

    # ------------------------------------------------------------------
    # Scrape ALL problems with checkpoint/resume
    # ------------------------------------------------------------------
    @staticmethod
    def _parse_rating(rating) -> Optional[int]:
        if isinstance(rating, int):
            return rating
        if isinstance(rating, str) and rating.isdigit():
            return int(rating)
        return None

    def scrape_all(self, output_file: str, checkpoint_file: str,
                   max_problems: Optional[int] = None,
                   skip_editorial: bool = False,
                   skip_solutions: bool = False,
                   min_rating: Optional[int] = None,
                   max_rating: Optional[int] = None):
        """
        Scrape all Codeforces problems. Automatically resumes from checkpoint.

        Output is JSONL (one JSON object per line) for streaming writes.
        """
        problems = self.fetch_problem_list()

        # Load checkpoint (already-scraped problem IDs)
        seen = set()
        if os.path.exists(checkpoint_file):
            with open(checkpoint_file) as f:
                seen = set(line.strip() for line in f if line.strip())
            logger.info("Checkpoint: %d problems already scraped", len(seen))

        # Filter & sort
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
            if min_rating is not None and (rating is None or rating < min_rating):
                continue
            if max_rating is not None and (rating is not None and rating > max_rating):
                continue

            filtered.append((pid, rating))

        if max_problems is not None:
            filtered = filtered[:max_problems]

        total = len(filtered)
        logger.info("Problems to scrape: %d (skipping %d already done)",
                     total, len(seen))

        if not filtered:
            logger.info("Nothing to scrape – all done!")
            return

        # Ensure output directory exists
        os.makedirs(os.path.dirname(output_file) or ".", exist_ok=True)

        success = 0
        errors = 0
        for i, (pid, api_rating) in enumerate(filtered):
            logger.info("[%d/%d] %s (rating=%s)", i + 1, total, pid,
                        api_rating or "?")
            try:
                data = self.scrape_problem(
                    pid,
                    api_rating=api_rating,
                    skip_editorial=skip_editorial,
                    skip_solutions=skip_solutions,
                )
                success += 1
                logger.info("  ✓ done (%d chars statement, %d chars editorial, "
                            "%d solutions)",
                            len(data.get("problem_statement", "")),
                            len(data.get("editorial_text", "")),
                            len(data.get("accepted_solutions", [])))
            except Exception as exc:
                logger.error("  ✗ failed: %s", exc)
                data = {"id": pid, "source": "Codeforces", "error": str(exc)}
                errors += 1

            # Append to JSONL (one line per problem – safe for streaming)
            with open(output_file, "a") as f:
                f.write(json.dumps(data, default=str) + "\n")
            # Mark as done in checkpoint
            with open(checkpoint_file, "a") as f:
                f.write(pid + "\n")

        logger.info(
            "="*60 + "\n"
            "Scraping complete: %d succeeded, %d failed out of %d\n"
            "Output: %s\n"
            "Checkpoint: %s\n" + "="*60,
            success, errors, total, output_file, checkpoint_file
        )

    # ------------------------------------------------------------------
    # Convert JSONL output to a single JSON array file
    # ------------------------------------------------------------------
    @staticmethod
    def jsonl_to_json(jsonl_path: str, json_path: str):
        """Convert JSONL output to a single pretty JSON array."""
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
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    parser = argparse.ArgumentParser(
        description="Scrape Codeforces problems into JSON.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s --all                          Scrape everything (resumes automatically)
  %(prog)s --all --max 10                 Scrape 10 problems (for testing)
  %(prog)s --all --min-rating 1200        Only problems rated ≥ 1200
  %(prog)s --all --light                  Skip editorials & solutions (fast)
  %(prog)s CF-1560A CF-4B                 Scrape specific problems
  %(prog)s --convert                      Convert JSONL output to JSON array
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
        "--delay", type=float, default=REQUEST_DELAY,
        help=f"Seconds between requests (default: {REQUEST_DELAY})",
    )
    parser.add_argument(
        "--cookies", type=str,
        help="JSON file with Codeforces session cookies (for accessing "
             "submission source code)",
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
        help="Skip editorial & solutions (much faster, statement + metadata only)",
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

    # Default output
    if not args.output:
        args.output = str(DEFAULT_OUTPUT)

    # Convert mode
    if args.convert:
        jsonl_path = args.output
        json_path = jsonl_path.replace(".jsonl", ".json")
        if json_path == jsonl_path:
            json_path += ".json"
        CodeforcesScraper.jsonl_to_json(jsonl_path, json_path)
        return

    # Ensure data dir exists
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    # Load cookies
    cookies = {}
    if args.cookies:
        with open(args.cookies) as f:
            cookies = json.load(f)

    scraper = CodeforcesScraper(delay=args.delay, cookies=cookies)

    if args.all:
        scraper.scrape_all(
            output_file=args.output,
            checkpoint_file=args.checkpoint,
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
